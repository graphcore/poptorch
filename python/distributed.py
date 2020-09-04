# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import subprocess
import json
from .logging import logger


class Partition:
    def __init__(self, num_ipus, num_gcds=1, num_sync_replicas=1):
        self.num_ipus = num_ipus
        self.num_gcds = num_gcds
        self.num_sync_replicas = num_sync_replicas

    def __str__(self):
        return str(self.__dict__)


def _vipu_cli(*args):
    args = [str(a) for a in args]
    return subprocess.check_output(["vipu-cli", "-a", "--showjson"] +
                                   list(args),
                                   stderr=subprocess.PIPE).decode('utf-8')


class VirtualIpuManager:
    @staticmethod
    def isAvailable():
        try:
            version = _vipu_cli("--version")
            logger.debug(version)
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.debug(e)
            return False

    @staticmethod
    def numIpus():
        """The number of IPUs available on the system"""
        clusters = json.loads(_vipu_cli("list", "cluster"))
        assert len(clusters) == 1
        cluster = clusters[0]
        spec = cluster["spec"]
        domains = spec["ipu_link_domains"]
        assert len(domains) == 1
        return len(domains[0]["ipus"])

    @staticmethod
    def listPartitions():
        partitions = json.loads(_vipu_cli("list", "partition"))
        retval = {}
        for p in partitions:
            name = p["partition"]["id"]
            num_ipus = len(p["partition"]["spec"]["ipus"])
            num_gcds = p["partition"]["spec"]["num_gcds"]
            num_sync_replicas = p["partition"]["spec"]["num_replicas"]
            retval[name] = Partition(num_ipus, num_gcds, num_sync_replicas)
        return retval

    @staticmethod
    def deletePartition(name):
        output = _vipu_cli("delete", "partition", name)
        logger.debug(output)

    @staticmethod
    def createPartition(name, partition):
        assert isinstance(partition, Partition)
        output = _vipu_cli("create", "partition", name, "--size",
                           partition.num_ipus, "--num-gcds",
                           partition.num_gcds, "--gcd-sync-replicas",
                           partition.num_sync_replicas)
        logger.debug(output)

    @staticmethod
    def resetPartition(name):
        output = _vipu_cli("reset", "partition", name)
        logger.debug(output)
