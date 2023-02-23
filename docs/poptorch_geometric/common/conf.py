# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# Configuration file for the Sphinx documentation builder.
#
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import json
import sys
import datetime
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'Project'
author = 'Graphcore Ltd'

# The full version, including alpha/beta/rc tags
# Looks like html uses 'version' and latex uses 'release'
version = 'v0.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
]


def get_current_release() -> str:
    format_str = "{major}.{minor}"
    version_file = pathlib.Path(__file__).parents[3].resolve() / "version.json"
    return format_str.format(**json.load(open(version_file)))


SDK_RELEASE = get_current_release()
extlinks = {
    'tutorials-repo':
    (f'https://github.com/graphcore/tutorials/tree/sdk-release-{SDK_RELEASE}/%s',
     None),
    'github-poptorch':
    (f'https://github.com/graphcore/poptorch/tree/sdk-release-{SDK_RELEASE}/%s',
     None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

nitpick_ignore = [
    ('py:obj', 'num_nodes * (num_nodes - 1)'),
    ('py:obj', '0.0'),
    ('py:obj', '1'),
    ('py:obj', '2'),
    ('py:obj', 'graphs_mask'),
    ('py:obj', 'nodes_mask'),
    ('py:obj', 'edges_mask'),
    ('py:obj', 'True'),
    ('py:obj', 'False'),
    ('py:obj', 'num_nodes'),
    ('py:obj', 'num_edges'),
    ('py:obj', 'num_graphs'),
    ('py:obj', 'None'),
    ('py:obj', 'data_source'),
    ('py:obj', 'follow_batch'),
    ('py:obj', 'exclude_keys'),
    ('py:obj', 'batch_sampler'),
    ('py:obj', 'shuffle'),
    ('py:obj', 'Data'),
    ('py:class', 'bool'),
    ('py:class', 'dict'),
    ('py:class', 'int'),
    ('py:class', 'iterable'),
    ('py:class', 'optional'),
    ('py:class', 'str'),
    ('py:class', 'T_co'),
    ('py:class', 'Dataset'),
    ('py:class', 'ClusterData'),
    ('py:class', 'Sampler'),
    ('py:class', '..'),
    ('py:class', 'torch.Tensor'),
    ('py:class', 'torch.utils.data.dataset.Dataset'),
    ('py:class', 'torch.utils.data.sampler.Sampler'),
    ('py:class', 'torch_geometric.data.Batch'),
    ('py:class', 'torch_geometric.data.Data'),
    ('py:class', 'torch_geometric.data.Dataset'),
    ('py:class', 'torch_geometric.data.dataset.Dataset'),
    ('py:class', 'torch_geometric.data.HeteroData'),
    ('py:class', 'torch_geometric.loader.cluster.ClusterData'),
    ('py:class', 'torch_geometric.loader.ClusterData'),
    ('py:class', 'poptorch.AsynchronousDataAccessor'),
    # Enums already described in functions that use them
    ('py:class', 'poptorch.DataLoader'),
    ('py:class', 'poptorch.Dataset'),
    ('py:class', 'poptorch.Options'),
    ('py:class', 'bool'),
    ('py:class', 'dict'),
    ('py:class', 'int'),
    ('py:class', 'iterable'),
    ('py:class', 'optional'),
    ('py:class', 'str'),
    ('py:class', 'T_co'),
    ('py:class', 'torch.Tensor'),
    ('py:class', 'torch.dtype'),
    ('py:class', 'torch.nn.Module'),
    ('py:class', 'torch.optim.Optimizer'),
    ('py:class', 'torch.optim.optimizer.Optimizer'),
    ('py:class', 'torch.utils.data.Dataset'),
    ('py:class', 'torch.utils.data.sampler.Sampler'),
    # Enums already described in functions that use them
    ('py:class', 'poptorch.OutputMode'),
    ('py:class', 'poptorch.ConnectionType'),
    ('py:class', 'poptorch.HalfFloatCastingBehavior'),
    ('py:class', 'poptorch.MatMulSerializationMode'),
    ('py:class', 'poptorch.OverlapMode'),
    ('py:class', 'poptorch.ReductionType'),
    ('py:class', 'poptorch.SyncPattern'),
    ('py:class', 'poptorch.MeanReductionStrategy'),
    # Type hints
    ('py:data', 'typing.Optional'),
    ('py:data', 'typing.Callable'),
    ('py:class', 'typing.ForwardRef'),
]

# Define abbreviations for IPU-PODn names with subscripts
# These use non-breaking hyphens & spaces, so be careful if editing or adding new definitions
pod_sizes = [2**i for i in range(2, 14)] + ["N"]
pod_definitions = [
    f".. |POD{i}| replace:: IPU‑POD\\ :subscript:`{i}`" for i in pod_sizes
]
bow_definitions = [
    f".. |BOW{i}| replace:: Bow Pod\\ :subscript:`{i}`" for i in pod_sizes
]

rst_epilog = ("\n".join(pod_definitions) + "\n" + "\n".join(bow_definitions) +
              r"""
.. role:: raw-html(raw)
    :format: html

.. |POD| replace:: IPU‑POD
.. |BOW| replace:: Bow Pod
.. |newpage| raw:: latex

  \newpage

.. |LEGAL:EULA| replace::
    This software is made available under the terms of the `Graphcore End User
    License Agreement (EULA) <https://docs.graphcore.ai/en/latest/eula.html>`__
    and the
    `Graphcore Container License Agreement <https://docs.graphcore.ai/projects/container-license/en/latest/>`__.
    Please ensure you have read and accept the terms of the corresponding
    license before using the software. The Graphcore EULA applies unless
    indicated otherwise.

.. |LEGAL:TRADEMARkS| replace::
    Graphcloud®, Graphcore® and Poplar® are registered trademarks of Graphcore
    Ltd.
    :raw-html:`<br></br>`
    Bow™,
    Bow-2000™,
    Bow Pod™,
    Colossus™,
    In-Processor-Memory™,
    IPU-Core™,
    IPU-Exchange™,
    IPU-Fabric™,
    IPU-Link™,
    IPU-M2000™,
    IPU-Machine™,
    IPU-POD™,
    IPU-Tile™,
    PopART™,
    PopDist™,
    PopLibs™,
    PopRun™,
    PopVision™,
    PopTorch™,
    Streaming Memory™
    and Virtual-IPU™
    are trademarks of Graphcore Ltd.
    :raw-html:`<br></br>`
    All other trademarks are the property of their respective owners.

""" + f".. |YEAR| replace:: {datetime.date.today().year}" + "\n" +
              f".. |SDK_RELEASE| replace:: {SDK_RELEASE}"
              "\n")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {'logo_only': False, 'navigation_depth': 4}

numfig = True

numfig_format = {
    'section': 'Section {number}, {name}',
    'figure': 'Fig. %s',
    'table': 'Table %s',
    'code-block': 'Listing %s'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../../common/_static']

# CSS file to create the Graphcore style
html_css_files = [
    'css/custom_rtd.css',
]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'Document Title'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '../../common/graphcorelogo-html.png'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# -- Options for LaTeX output ---------------------------------------------
# Don't know how much of this is necessary. It's a bit of a mess.

# pifont required for tick and cross characters
# use array stretch to get taller table rows. Also consider sing extrarowheight
# \\setlength{\\extrarowheight}{1pt}

ADDITIONAL_PREAMBLE = r"""
\setcounter{secnumdepth}{5}
\setcounter{tocdepth}{5}

\usepackage{threeparttable}
\usepackage{pifont}
\usepackage{array}
\usepackage{charter}
\usepackage[defaultsans]{lato}
\usepackage{inconsolata}
\usepackage{listings}
\usepackage{verbatim}
\usepackage{multicol}
\usepackage{float}
\usepackage{fancyhdr}
%% Obtain access to ssmall font size
\usepackage[10pt]{moresize}

%% adjustbox used to set max width and height for images
\usepackage[export]{adjustbox}

%% Define a right-justified table column type
\usepackage{ragged2e}
\newcolumntype{R}[1]{>{\RaggedLeft\arraybackslash}p{#1}}

\renewcommand{\arraystretch}{1.4}

\usepackage{booktabs}
\usepackage{graphicx}

%% Push footnotes to the bottom of the page
\usepackage[bottom]{footmisc}

\usepackage{pdfpages}

\usepackage{pdflscape}

\usepackage{transparent}
\usepackage[normalem]{ulem}

%% Watermark stuff
\usepackage{draftwatermark}
\SetWatermarkFontSize{2cm}
\SetWatermarkColor[gray]{0.96}
\SetWatermarkText{}
\SetWatermarkScale{2}
\SetWatermarkAngle{30}

%% Ensure API descriptions are all tt family
\let\fulllineitemsOld\fulllineitems
\let\endfulllineitemsOld\endfulllineitems
\renewenvironment{fulllineitems}{\ttfamily\small\fulllineitemsOld}{\endfulllineitemsOld}

%% Change the Sphinx verbatim to not put the box around it and to indent
\renewcommand{\Verbatim}[1][1]{%
  % list starts new par, but we don't want it to be set apart vertically
  \bgroup\parskip=0pt%
  \medskip
  % The list environment is needed to control perfectly the vertical
  % space.
  \list{}{%
  \setlength\parskip{0pt}%
  \setlength\itemsep{0ex}%
  \setlength\topsep{0ex}%
  \setlength\partopsep{0pt}%
  \setlength\leftmargin{0pt}%
  }%
  \OriginalVerbatim[#1,xleftmargin=0.5cm,formatcom=\normalsize]%
}
\renewcommand{\endVerbatim}{%
    \endOriginalVerbatim%
  \endlist%
  % close group to restore \parskip
  \egroup%
}

\newcommand{\VerbBorders}{%
  \renewcommand{\Verbatim}[1][1]{%
    % list starts new par, but we don't want it to be set apart vertically
    \bgroup\parskip=0pt%
    \smallskip%
    % The list environment is needed to control perfectly the vertical
    % space.
    \list{}{%
      \setlength\parskip{0pt}%
      \setlength\itemsep{0ex}%
      \setlength\topsep{0ex}%
      \setlength\partopsep{0pt}%
      \setlength\leftmargin{0pt}%
    }%
    \item\MakeFramed {\FrameRestore}%
    \small%
    \OriginalVerbatim[##1]%
  }
  \renewcommand{\endVerbatim}{%
    \endOriginalVerbatim%
    \endMakeFramed%
    \endlist%
    % close group to restore \parskip
    \egroup%
  }
  \definecolor{VerbatimColor}{rgb}{0.95,0.95,0.95}
  \definecolor{VerbatimBorderColor}{rgb}{1.0,1.0,1.0}
}

\makeatletter
\DeclareTextCommandDefault{\textleftarrow}{\mbox{$\m@th\leftarrow$}}
\makeatother
"""

ADDITIONAL_PREAMBLE += r"""
%% Redefine sphinxstylethead (used only for table headers) to bold font
\usepackage{letltxmacro}
\LetLtxMacro{\oldtextsf}{\sphinxstyletheadfamily}
\renewcommand{\sphinxstyletheadfamily}[0]{\oldtextsf \bf }
"""

ADDITIONAL_PREAMBLE += r"""
\makeatletter
  \fancypagestyle{normal}{
    \fancyhf{}
    \fancyfoot[RE,RO]{{\py@HeaderFamily\thepage}}
    \fancyfoot[LE,LO]{%(footer)s}
    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0.4pt}
  }
  \fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[RE,RO]{{\py@HeaderFamily\thepage}}
    \fancyfoot[LE,LO]{%(footer)s}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0.4pt}
  }
\makeatother
""" % {
    'footer': ''
}

# From Sphinx 1.5 onwards, there are certain macros which are used which became
# deprecated (e.g. \code). These macros should be upgraded in the future so
# that we can move away from using the old macro names.
latex_keep_old_macro_names = True

latex_elements = {
    # Options to pass to packages
    'passoptionstopackages':
    r'\PassOptionsToPackage{dvipsnames, table, xcdraw}{xcolor}',

    # Set up margins for geometry
    'sphinxsetup': 'hmargin={0.75in, 0.75in}, vmargin={0.75in, 0.75in}',

    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',

    # Single sided to save paper and improve display
    'extraclassoptions': 'openany,oneside',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',

    # Disable figure floating
    'figure_align': 'H',

    # Additional stuff for the LaTeX preamble.
    'preamble': ADDITIONAL_PREAMBLE,
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_class = 'manual'
if 'DOC_TITLE' in os.environ:
    latex_title = os.environ['DOC_TITLE']
else:
    latex_title = "Document title"
latex_documents = [
    ('index', 'doc.tex', latex_title, author, latex_class),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = '../../common/graphcorelogo-pdf.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = False

# If true, show page references after internal links.
latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
latex_domain_indices = False

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_default_options = {
    'undoc-members': True,
}
autodoc_inherit_docstrings = True

autodoc_typehints = 'description'
