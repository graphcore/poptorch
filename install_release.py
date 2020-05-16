# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
# This script is run by the release agent to create a release of PopTorch


def install_release(release_utils, release_id, snapshot_id, version_str):
    release_utils.log.info('Tagging poptorch release ' + version_str)

    # Create the release on the document server.
    release_utils.create_document_release(snapshot_id)

    # Tag the view repository with the release.
    release_utils.tag_view_repo(
            'ssh://git@phabricator.sourcevertex.net/diffusion/' \
            + 'POPTORCH/poptorch.git',
            snapshot_id,
            release_id,
            version_str)

    # Increment the point version number.
    release_utils.increment_version_point(
            'ssh://git@phabricator.sourcevertex.net/diffusion/' \
            + 'POPTORCH/poptorch.git')
