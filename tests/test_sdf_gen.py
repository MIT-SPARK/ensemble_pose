import copy
import logging
import pathlib
import tempfile

from utils.sdf_gen.mesh2sdf_floodfill import vox


def test_vox_eq():
    logging.basicConfig(level=logging.DEBUG)
    test_data_folder_path = pathlib.Path(__file__).parent.resolve().joinpath("./test_data/sdf_gen")
    # load vox file
    vox_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_sdf.vox")
    vox_data = vox.load_vox_sdf(vox_path.__str__())
    vox_data_empty = vox.VoxSDF()
    assert vox_data != vox_data_empty

    vox_data_2 = vox.load_vox_sdf(vox_path.__str__())
    assert vox_data == vox_data_2

    vox_data_3 = copy.deepcopy(vox_data)
    assert vox_data == vox_data_3


def test_vox_loading_and_writing():
    logging.basicConfig(level=logging.DEBUG)
    test_data_folder_path = pathlib.Path(__file__).parent.resolve().joinpath("./test_data/sdf_gen")
    tmp_dir = tempfile.gettempdir()
    # load vox file
    vox_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_sdf.vox")
    vox_data = vox.load_vox_sdf(vox_path.__str__())

    # write to vox file
    vox2_path = tmp_dir.joinpath("temp.vox").__str__()
    vox.write_vox_sdf(vox2_path, vox_data)

    # read the written vox again
    vox_data_reloaded = vox.load_vox_sdf(vox2_path.__str__())

    # compare with data loaded from the first time
    logging.debug(f"Original Vox Data: \n{vox_data}")
    logging.debug(f"Re-loaded Vox Data: \n{vox_data_reloaded}")
    assert vox_data_reloaded == vox_data
