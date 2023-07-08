export SRC=https://bop.felk.cvut.cz/media/data/bop_datasets

# lm
wget $SRC/lm_base.zip         # Base archive with dataset info, camera parameters, etc.
wget $SRC/lm_models.zip       # 3D object models.
wget $SRC/lm_test_all.zip     # All test images ("_bop19" for a subset used in the BOP Challenge 2019/2020).
wget $SRC/lm_train_pbr.zip    # PBR training images (rendered with BlenderProc4BOP).

# lm-o
wget $SRC/lmo_base.zip
wget $SRC/lmo_models.zip
wget $SRC/lmo_train_pbr.zip
wget $SRC/lmo_train.zip
wget $SRC/lmo_test_all.zip
wget $SRC/lmo_test_bop19.zip

# tless
wget $SRC/tless_base.zip
wget $SRC/tless_models.zip
wget $SRC/tless_train_pbr.zip
wget $SRC/tless_train_render_reconst.zip
wget $SRC/tless_train_primesense.zip
wget $SRC/tless_test_primesense_all.zip
wget $SRC/tless_test_primesense_bop19.zip

# itodd
wget $SRC/itodd_base.zip
wget $SRC/itodd_models.zip
wget $SRC/itodd_train_pbr.zip
wget $SRC/itodd_val.zip
wget $SRC/itodd_test_all.zip
wget $SRC/itodd_test_bop19.zip

# hb
wget $SRC/hb_base.zip
wget $SRC/hb_models.zip
wget $SRC/hb_train_pbr.zip
wget $SRC/hb_val_primesense.zip
wget $SRC/hb_val_kinect.zip
wget $SRC/hb_test_primesense_all.zip
wget $SRC/hb_test_kinect_all.zip
wget $SRC/hb_test_bop19.zip

# hope
wget $SRC/hope_base.zip
wget $SRC/hope_models.zip
wget $SRC/hope_val.zip
wget $SRC/hope_test_all.zip
wget $SRC/hope_test_bop19.zip

# ycbv
wget $SRC/ycbv_base.zip
wget $SRC/ycbv_models.zip
wget $SRC/ycbv_train_pbr.zip
wget $SRC/ycbv_train_synt.zip
wget $SRC/ycbv_train_real.zip
wget $SRC/ycbv_test_all.zip
wget $SRC/ycbv_test_bop19.zip

# ru-apc
wget $SRC/ruapc_base.zip
wget $SRC/ruapc_models.zip
wget $SRC/ruapc_train.zip
wget $SRC/ruapc_test_all.zip
wget $SRC/ruapc_test_bop19.zip

# ic-bin
wget $SRC/icbin_base.zip
wget $SRC/icbin_models.zip
wget $SRC/icbin_train.zip
wget $SRC/icbin_train_pbr.zip
wget $SRC/icbin_test_all.zip
wget $SRC/icbin_test_bop19.zip

# ic-mi
wget $SRC/icmi_base.zip
wget $SRC/icmi_models.zip
wget $SRC/icmi_train.zip
wget $SRC/icmi_test_all.zip
wget $SRC/icmi_test_bop19.zip

# tudl
wget $SRC/tudl_base.zip
wget $SRC/tudl_models.zip
wget $SRC/tudl_train_pbr.zip
wget $SRC/tudl_train_render.zip
wget $SRC/tudl_train_real.zip
wget $SRC/tudl_test_all.zip
wget $SRC/tudl_test_bop19.zip

# tyo-l
wget $SRC/tudl_base.zip
wget $SRC/tudl_models.zip
wget $SRC/tudl_test_all.zip
wget $SRC/tudl_test_bop19.zip
