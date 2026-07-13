



def diffusion_models(task_model,net_param,**kwargs):

    if task_model=="DiffSTG":
        from models.Diffusion_model.DiffSTG.graph_diffusion_model import DiffSTG
        return DiffSTG(net_param=net_param)
    elif task_model=="DiffusionTS":
        from models.Diffusion_model.DiffusionTS.DiffusionTS_model import DiffusionTS_model
        return DiffusionTS_model(net_param=net_param)
    elif task_model=="TMDM":
        from models.Diffusion_model.TMDM.tmdm_adapter import TMDM_model
        return TMDM_model(net_param=net_param)
    elif task_model=="NsDiff_spatial":
        from models.Diffusion_model.NsDiff.NsDiff_model import NsDiff_model_spatial
        return NsDiff_model_spatial(net_param=net_param,train_model_select=kwargs["train_model_select"],
                                    pretrain_f_path=net_param["pretrain_f_path"] if net_param.get("pretrain_f_path") else None,
                                    pretrain_g_path=net_param["pretrain_g_path"] if net_param.get("pretrain_g_path") else None,)
    elif task_model=="NsDiff":
        from models.Diffusion_model.NsDiff.NsDiff_model    import NsDiff_model
        return NsDiff_model(net_param=net_param,train_model_select=kwargs["train_model_select"],
                            pretrain_f_path=net_param["pretrain_f_path"] if net_param.get("pretrain_f_path") else None,
                            pretrain_g_path=net_param["pretrain_g_path"] if net_param.get("pretrain_g_path") else None,)
    elif task_model=="NsDiff_model_variants":
        from models.Diffusion_model.NsDiff.NsDiff_model    import NsDiff_model_variants
        return NsDiff_model_variants(net_param=net_param,train_model_select=kwargs["train_model_select"],
                            )
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")


