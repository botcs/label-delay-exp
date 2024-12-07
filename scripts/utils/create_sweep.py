import wandb

sweep_configuration = {
    "name": "SSL Methods loss-scaling grid [1e-3 - 1e-0]",
    "command": [
        "python",
        "main_delay.py",
        "--config-path=scripts/custom",
        "${args_no_hyphens}",
        "data.cache_num_images=3840000",
        "devices=[0]",
        "online.delay=50",
        "+limit_train_batches=10000"
    ],
    "metric": {"name": "next_batch_acc1_epoch", "goal": "maximize"},
    "method": "grid",
    "parameters": {
        "--config-name": {
            "values": [
                "cloc_deepclusterv2.yaml",
                "cloc_byol.yaml",
                "cloc_nnbyol.yaml",
                "cloc_simsiam.yaml",
                "cloc_nnsiam.yaml",
                "cloc_wmse.yaml",
                "cloc_barlow.yaml",
                "cloc_dino.yaml",
                "cloc_mocov2plus.yaml",
                "cloc_mocov3.yaml",
                "cloc_nnclr.yaml",
                "cloc_ressl.yaml",
                "cloc_supcon.yaml",
                "cloc_swav.yaml",
                "cloc_vibcreg.yaml",
                "cloc_vicreg.yaml",
            ]
        },
        "method_kwargs.scale_loss": {"values": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    }
} 

