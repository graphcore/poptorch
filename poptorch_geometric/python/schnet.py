# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn.models.schnet import (GaussianSmearing,
                                              InteractionBlock,
                                              ShiftedSoftplus)
from torch_scatter.scatter import scatter_add

from poptorch import identity_loss


class SchNet(nn.Module):
    def __init__(self,
                 num_features=128,
                 num_interactions=2,
                 num_gaussians=50,
                 cutoff=6.0,
                 batch_size=None):
        """
        :param num_features (int): The number of hidden features used by both
            the atomic embedding and the convolutional filters (default: 128).
        :param num_interactions (int): The number of interaction blocks
            (default: 2).
        :param num_gaussians (int): The number of gaussians used in the radial
            basis expansion (default: 50).
        :param cutoff (float): Cutoff distance for interatomic interactions
            which must match the one used to build the radius graphs
            (default: 6.0).
        :param batch_size (int, optional): The number of molecules in the
            batch. This can be inferred from the batch input when not supplied.
        """
        super().__init__()
        self.num_features = num_features
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.batch_size = batch_size

        self.atom_embedding = nn.Embedding(100,
                                           self.num_features,
                                           padding_idx=0)
        self.basis_expansion = GaussianSmearing(0.0, self.cutoff,
                                                self.num_gaussians)

        self.interactions = nn.ModuleList()

        for _ in range(self.num_interactions):
            block = InteractionBlock(self.num_features, self.num_gaussians,
                                     self.num_features, self.cutoff)
            self.interactions.append(block)

        self.lin1 = nn.Linear(self.num_features, self.num_features // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(self.num_features // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize learnable parameters used in training the SchNet model.
        """
        self.atom_embedding.reset_parameters()

        for interaction in self.interactions:
            interaction.reset_parameters()

        xavier_uniform_(self.lin1.weight)
        zeros_(self.lin1.bias)
        xavier_uniform_(self.lin2.weight)
        zeros_(self.lin2.bias)

    def forward(self, z, edge_weight, edge_index, batch, target=None):
        """
        Forward pass of the SchNet model

        :param z: Tensor containing the atomic numbers for each atom in the
            batch. Vector of integers with size [num_atoms].
        :param edge_weight: Tensor containing the interatomic distances for
            each interacting pair of atoms in the batch. Vector of floats with
            size [num_edges].
        :param edge_index: Tensor containing the indices defining the
            interacting pairs of atoms in the batch. Matrix of integers with
            size [2, num_edges]
        :param batch: Tensor assigning each atom within a batch to a molecule.
            This is used to perform per-molecule aggregation to calculate the
            predicted energy. Vector of integers with size [num_atoms]
        :param target (optional): Tensor containing the target to
            use for evaluating the mean-squared-error loss when training.
        """
        # Collapse any leading batching dimensions
        z = z.view(-1)
        edge_weight = edge_weight.view(-1)
        edge_index = edge_index.view(2, -1)

        h = self.atom_embedding(z)
        edge_attr = self.basis_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # zero out embeddings for padding atoms
        mask = (z == 0).view(-1, 1)
        h = h.masked_fill(mask.expand_as(h), 0.)

        batch = batch.view(-1)
        out = scatter_add(h, batch, dim=0, dim_size=self.batch_size).view(-1)

        if not self.training:
            return out

        target = target.view(-1)
        loss = self.loss(out, target)
        return out, loss

    @staticmethod
    def loss(input, target):
        """
        Calculates the mean squared error

        This loss assumes that zeros are used as padding on the target so that
        the count can be derived from the number of non-zero elements.
        """
        loss = F.mse_loss(input, target, reduction="sum")
        N = (target != 0.0).to(loss.dtype).sum()
        loss = loss / N
        return identity_loss(loss, reduction="none")
