import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.donnees_simulees import creer_artefacts  # noqa: E402


@pytest.fixture(scope="session", params=[False, True], ids=["mcc_categoriel", "mcc_numerique"])
def artefacts(tmp_path_factory, request):
    dossier = tmp_path_factory.mktemp(f"artefacts_{request.param}")
    brut, var = creer_artefacts(dossier, utilise_num=request.param)
    return dossier, brut, var
