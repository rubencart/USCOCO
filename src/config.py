import logging
import random
from types import MethodType
from typing import Any, List, Literal, Optional

import torch.cuda
from tap import Tap

logger = logging.getLogger("pytorch_lightning")


class ModelConfig(Tap):
    length_dropout: float = 0.0

    length_loss_coef: float = 0.1
    length_label_smoothing: float = 0.05
    weight_length_loss_by_dist: bool = True

    ignore_eos_loss: bool = True
    max_target_positions: int = 96
    predict_num_queries: bool = False
    train_text_encoder: bool = False
    encoder_embed_dim: int = 256
    use_additional_detr_encoder: bool = True
    detr_encoder_for_length_only: bool = False

    no_pos_encoding: bool = True
    pos_first_layer_only: bool = False  # can only be True if no_pos_encoding = False
    no_fc_after_text_encoder: bool = False

    obj_gan: bool = False
    autoregressive: bool = False
    decode_max_objects: int = -1

    sequence_encoder_bias: bool = True


class TreeConfig(Tap):
    n: int = 2
    k: int = 16
    p_repeats: int = 16
    max_nodes: int = 80
    node_feature_size: int = 512
    seq_pos_encoding: bool = False
    encoder_nb_att_layers: int = 4


class DETRConfig(Tap):
    dropout: float = 0.1
    dropout_bbox_embed: float = 0.1
    dropout_token_embed: float = 0.1

    encoder_ffn_embed_dim: int = 1024
    encoder_layers: int = 4
    encoder_attention_heads: int = 8

    decoder_embed_dim: int = 256
    decoder_ffn_embed_dim: int = 1024
    decoder_layers: int = 6
    decoder_attention_heads: int = 8

    normalize_prenorm: bool = False

    aux_loss: bool = False
    recompute_indices_for_aux_losses: bool = True
    probabilistic_bbox_predictions: bool = True
    bbox_regression_layers: int = 2
    label_pos_head_ffn_dim: int = 1024
    class_project_mlp: bool = True

    tie_label_embed_to_bbox_label_embed: bool = False
    tie_label_embed_to_label_output_proj: bool = False
    tie_bbox_embed_to_bbox_output_proj: bool = False
    # or 'probs', 'embed', 'logits', 'softmax_embed', None
    condition_bbox_prediction_on_label: Optional[str] = None
    softmax_embed_training_argmax_inference: bool = True

    noise_as_queries: bool = False
    learnt_query_embeds: bool = True
    query_pos_first_layer_only: bool = False
    query_noise_aggregate: str = "sum"  # or 'concat'

    lr: float = 1.0e-4
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_eps: float = 1e-6
    weight_decay: float = 0.01

    num_queries: int = 96
    activation: str = "relu"

    set_cost_class_coef: float = 1
    set_cost_bbox_coef: float = 1
    set_cost_giou_coef: float = 0.4
    set_cost_prob_bbox_coef: float = 0
    struct_loss_input: Literal["encoder_out", "tree_pos_embs"] = "tree_pos_embs"
    load_span_tree_pos_embs: bool = False
    struct_loss_coef: float = 0.0
    label_loss_coef: float = 0.5
    bbox_loss_coef: float = 5
    giou_loss_coef: float = 2
    bbox_prop_loss_coef: float = 0.5
    bbox_rel_loss_coef: float = 0.5
    bbox_rel_loss_type: str = "diag_scaled"  # abs | diag_scaled
    prob_bbox_loss_coef: float = 1
    nobj_coef: float = 0.1

    struct_loss_type: Literal["max", "attn"] = "max"
    struct_loss_margin: float = 1.0
    struct_loss_gamma1: float = 5.0
    struct_loss_gamma2: float = 5.0
    struct_loss_gamma3: float = 10.0
    struct_loss_normalize_sim_scores_across_spans: bool = True
    struct_loss_separate_sent: bool = False

    # autoregressive
    label_smoothing: float = 0.05
    label_smoothing_pos: float = 0.05
    weight_prob_bbox_loss_by_dist: bool = True
    min_num_generated_objects: int = 2  # 1 (bos) + 1 obj?
    max_num_generated_objects: int = 24

    generation_strategy: Literal["nucleus", "greedy", "sample", "detr"] = "greedy"
    generation_strategy_for_eval_criterion: Literal["nucleus", "greedy", "sample", "detr"] = (
        "greedy"
    )
    # generation_strategies: List[str] = ['nucleus', 'sample', 'greedy']      # for AR model
    generation_strategies: List[str] = ["greedy"]  # for AR model

    # inherit takes from label generation strategy
    bbox_generation_strategy: Literal["nucleus", "greedy", "sample", "detr", "inherit"] = "greedy"
    bbox_pred_hard_sigmoid: bool = False

    # https://arxiv.org/abs/1904.09751
    generation_nucleus_sample_top_p: float = 0.95
    generation_temperature: float = 1.0

    label_bbox_embed_aggregate: str = "sum"  # or 'concat_half' , 'concat_proj'
    bbox_embed_aggregate: str = "sum"  # or 'concat'


class ObjGANConfig(Tap):
    gmm_comp_num: int = 5
    input_dim: int = 300
    hidden_dim: int = 128
    box_hidden_dim: int = 50
    n_layers: int = 1
    rnn_cell: str = "lstm"
    bidirectional_enc: bool = True

    temperature: float = 0.4
    input_dropout_p: float = 0.5
    dropout_p: float = 0.5  # 0.5
    use_attention: bool = True
    mean_std_path: str = "./data/mean_std_train2017.json"
    # 128.315624 65.104890
    # 132.679686 53.485141
    # 44.473426 54.300979
    # 1.924934 1.642659

    bbox_loss_coef: float = 1.0
    label_loss_coef: float = 1.0

    gaussian_dict_path: str = "./data/gaussian_dict.npy"


class TextEncoderConfig(Tap):
    text_encoder: Literal[
        "huggingface",
        "sent_clip",
        "vokenization",
        "attn_gan",
        "plm",
        "gpt2_bllip",
        "tg",
    ] = "huggingface"
    txt_enc_pretrained: bool = True
    txt_enc_finetune: bool = False
    num_words: int = 40

    # visualbert
    # hf_tokenizer_model_name_or_path: str = 'bert-base-uncased'
    # hf_model_name_or_path: str = 'uclanlp/visualbert-vqa-coco-pre'

    # clip
    clip_model_name: str = "ViT-B/32"

    # huggingface
    hf_model_type: str = "roberta"
    hf_config_name: str = ""
    hf_model_name_or_path: str = "distilroberta-base"
    hf_tokenizer_model_name_or_path: Optional[str] = None
    cache_dir: str = "cache"
    huggingface_offline: bool = False
    hf_finetune_last_n_layers: int = -1
    use_llama: bool = False

    add_len_token: bool = False
    sort_by_caption_len: bool = False

    lr: float = 5e-5
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_eps: float = 1e-6
    weight_decay: float = 0.001

    attn_gan_text_encoder_input_dim: int = 1560
    attn_gan_text_encoder_hidden_dim: int = 1560
    # todo
    attn_gan_text_encoder_path = "./models/AttnGAN-lrg.pth"
    attn_gan_vocab_path: str = "./data/vpcfg_coco.dict.pkl"
    attn_gan_text_encoder_dropout: float = 0.5
    attn_gan_text_encoder_nlayers: int = 2
    attn_gan_text_encoder_bidirectional: bool = True

    attn_gan_extra_encoder_nb_att_layers: int = 0
    attn_gan_extra_encoder_hidden_size: int = 256
    attn_gan_extra_encoder_dropout: float = 0.1
    attn_gan_extra_encoder_att_heads: int = 8
    attn_gan_extra_encoder_att_dff: int = 2048
    attn_gan_extra_encoder_activation: str = "relu"
    attn_gan_extra_encoder_normalize_before: bool = False

    txt_enc_d_model: int = 512
    txt_enc_dim_feedforward: int = 2048
    txt_enc_nhead: int = 8
    txt_enc_num_layers: int = 8
    txt_enc_dropout: float = 0.1

    # todo
    # PLM
    plm_checkpoint: str = "./models/pretrained_text_encoders/xplm_bllip-lg_rand-init_1101_5.params"
    # in case you want to use the PLM_mask model, uncomment the next line and set
    #   plm_add_structured_mask to True
    # plm_checkpoint: str = './models/pretrained_text_encoders/xplm-mask_bllip-lg_rand-init_1101_5.params'
    plm_add_structured_mask: bool = False

    # GPT-2_Bllip small (the lg concerns the bllip dataset)
    # lm_checkpoint: str = "./models/pretrained_text_encoders/xlm_bllip-lg_rand-init_1103_5.params"
    lm_checkpoint: str = "./models/pretrained_text_encoders/xlm_bllip-lg_rand-init_1101_5.params"
    # lm_checkpoint: str = './models/pretrained_text_encoders/lm_gpt2-medium.params'
    # lm_checkpoint: str = './models/pretrained_text_encoders/lm_gpt2-large.params'

    # TG
    # For right branching, uncomment the next line and set tg_right_branching to True
    # tg_checkpoint: str = './models/pretrained_text_encoders/rightBranch_tg_37927.params'
    tg_right_branching: bool = False

    # TG with gpt2-small (?):
    tg_checkpoint: str = "./models/pretrained_text_encoders/tg_37927.params"
    # tg_checkpoint: str = './models/pretrained_text_encoders/tg_37928.params' # gpt2-large
    # tg_checkpoint: str = './models/pretrained_text_encoders/tg_md_37928.params' # gpt2-medium

    architecture: str = "gpt2"  # gtp2, gpt2-medium, gpt2-large


class TrainConfig(Tap):
    accelerator: str = "gpu"
    max_epochs: int = 2
    min_epochs: int = 0
    max_steps: int = -1
    precision: int = 32
    gradient_clip_val: float = 3.0
    gpus: Any = None


class ProbeConfig(Tap):
    """
    https://arxiv.org/pdf/1905.06316.pdf page 15:
    With the exception of ELMo scalars, we hold the weights of the sentence encoder (§ 3.2) fixed
    while we train our probing classifier. We train using the Adam optimizer (Kingma & Ba, 2015)
    with a batch size9 of 32, an initial learning rate of 1e-4, and gradient clipping with
    max L2 norm
    of 5.0. We evaluate on the validation set every 1000 steps (or every 100 for SPR1, SPR2, and
    Winograd), halve the learning rate if no improvement is seen in 5 validations,
    and stop training if no
    improvement is seen in 20 validations.
    """

    # TG:
    # /cw/working-arwen/rubenc/MMMAli/output/2022_10_03_16_21_45_detr_s-41_TG/test_run/probe_embeddings.h5
    # TG attn 0.25:
    # /cw/working-arwen/rubenc/MMMAli/output/2022_09_28_13_01_57_detr_s-41_detr-TG-struct_attn25/test_run/probe_embeddings.h5
    # TG RB:
    # /cw/working-theoden/rubenc/MMMAli/output/2023_03_30_22_49_13_detr_s-43_TG-rb/test_run/probe_embeddings.h5
    # TG RB attn 0.50:
    # /cw/working-theoden/rubenc/MMMAli/output/2023_04_07_10_40_37_detr_s-42_TG-rb-attn05/test_run/probe_embeddings.h5
    # TG attn 0.50:
    # /cw/working-arwen/rubenc/MMMAli/output/2022_09_28_13_01_57_detr_s-42_detr-TG-struct_attn05/test_run/probe_embeddings.h5
    # Qian base:
    # /cw/working-arwen/rubenc/MMMAli/output/2022_09_28_21_17_33_detr_s-41_qian-base/test_run/probe_embeddings.h5
    # Qian base attn 0.5:
    # /cw/working-arwen/rubenc/MMMAli/output/2022_09_28_21_10_40_detr_s-41_qian-base-attn05/test_run/probe_embeddings.h5

    # GPT-2:
    # /cw/working-arwen/rubenc/MMMAli/output/2022_10_03_16_29_36_detr_s-43_gpt2/test_run/probe_embeddings.h5
    # GPT-2 attn 0.5:
    # /cw/working-arwen/rubenc/MMMAli/output/2022_10_03_17_05_50_detr_s-42_gpt2-attn25/test_run/probe_embeddings.h5

    # GPT-2 lg attn 0.5:
    # /cw/working-theoden/rubenc/MMMAli/output/2023_02_28_17_10_23_detr_s-43_gpt2-lg-attn05/test_run/probe_embeddings.h5
    # GPT-2 lg:
    # /cw/working-theoden/rubenc/MMMAli/output/2023_02_23_17_34_43_detr_s-42_gpt2-lg/test_run/probe_embeddings.h5
    # GPT-2 md attn 0.25:
    # /cw/working-theoden/rubenc/MMMAli/output/2023_02_23_18_08_41_detr_s-42_gpt2-md-attn/test_run/probe_embeddings.h5
    # GPT-2 md:
    # /cw/working-theoden/rubenc/MMMAli/output/2023_02_23_17_34_29_detr_s-41_gpt2-md/test_run/probe_embeddings.h5

    # '/cw/working-arwen/rubenc/MMMAli/output/2022_10_03_23_43_52_detr_s-42_qian-mask-attn25/test_run/probe_embeddings.h5'
    # '/cw/working-arwen/rubenc/MMMAli/output/2022_10_03_16_22_12_detr_s-42_qian-mask/test_run/probe_embeddings.h5'
    # '/cw/working-arwen/rubenc/MMMAli/output/2022_10_03_16_21_06_detr_s-43_qian/test_run/probe_embeddings.h5'
    # '/cw/working-arwen/rubenc/MMMAli/output/2022_09_28_11_13_01_detr_s-42_qian-struct_attn25/test_run/probe_embeddings.h5'

    # '/cw/working-arwen/rubenc/MMMAli/output/2022_10_03_23_43_52_detr_s-42_qian-mask-attn25/test_run/id_to_h5_idx.json',
    # '/cw/working-arwen/rubenc/MMMAli/output/2022_10_03_16_22_12_detr_s-42_qian-mask/test_run/id_to_h5_idx.json',
    # '/cw/working-arwen/rubenc/MMMAli/output/2022_10_03_16_21_06_detr_s-43_qian/test_run/id_to_h5_idx.json',
    # '/cw/working-arwen/rubenc/MMMAli/output/2022_09_28_11_13_01_detr_s-42_qian-struct_attn25/test_run/id_to_h5_idx.json'

    # 2023_04_26_09_25_46_detr_s-43_TG-lg
    # /cw/working-frodo/rubenc/MMMAli/output/2023_05_03_18_07_42_detr_s-44_TG-lg-attn25

    # 2023_04_24_15_25_46_detr_s-42_detr-llama
    # 2023_04_26_10_36_06_detr_s-43_detr-llama30B

    # /cw/working-arwen/rubenc/MMMAli/output/2023_04_27_18_15_35_detr_s-43_qian-base-lg
    # /cw/working-arwen/rubenc/MMMAli/output/2023_04_27_18_15_35_detr_s-43_qian-base-lg-attn50

    # /cw/working-arwen/rubenc/MMMAli/output/2022_10_13_08_42_23_detr_s-41_detr-qian-base-shuffle

    # todo
    h5_path: str = "./output/???/test_run/probe_embeddings.h5"
    h5_index_path: str = "./output/???/test_run/id_to_h5_idx.json"
    # embedding_model: Literal['TG', 'LM'] = 'TG'

    learn_negative_constituents: bool = True
    learn_which_negative_constituents: Literal["all", "as_positives"] = "as_positives"
    eval_which_negative_constituents: Literal["all", "as_positives"] = "as_positives"
    learn_tags: bool = True
    tg_include_parens: bool = True
    tg_harder_negatives: bool = False

    batch_size: int = 128
    dropout: float = 0.5
    lr: float = 1e-4
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_eps: float = 1e-6
    weight_decay: float = 0.001

    lr_schedule: Literal["linear_with_warmup", "reduce_on_plateau"] = "reduce_on_plateau"  # or None
    lr_schedule_monitor: str = "val_sent_micro_tag_macro_f1"
    lr_schedule_mode: str = "max"
    lr_schedule_factor: float = 0.5
    lr_schedule_patience: int = 3
    val_every_n_steps: float = 1.0  # every epoch

    current_layer: int = 0
    start_at_layer: int = 0
    loop_over_layers: bool = False


class Config(Tap):
    dataset: str = "coco17"
    captions: str = "coco"
    # todo
    wandb_project_name: str = "USCOCO"
    wandb_org_name = "liir-kuleuven"
    save_probe_embeddings: bool = False
    save_probe_embeddings_train: bool = False

    # todo
    train_image_dir: str = "/cw/liir_data/NoCsBack/MSCOCO/Images/2017/train2017"
    val_image_dir: str = "/cw/liir_data/NoCsBack/MSCOCO/Images/2017/val2017"
    # train_image_dir: str = "./data/val2017"
    # val_image_dir: str = "./data/val2017"

    # todo
    train_captions: str = "/cw/liir_data/NoCsBack/MSCOCO/annotations/captions_train2017.json"
    val_captions: str = "/cw/liir_data/NoCsBack/MSCOCO/annotations/captions_val2017.json"
    # train_captions: str = "./data/captions_val2017.json"
    # val_captions: str = "./data/captions_val2017.json"

    # todo
    train_instances: str = "/cw/liir_data/NoCsBack/MSCOCO/annotations/instances_train2017.json"
    val_instances: str = "/cw/liir_data/NoCsBack/MSCOCO/annotations/instances_val2017.json"
    # train_instances: str = "./data/instances_val2017.json"
    # val_instances: str = "./data/instances_val2017.json"

    uscoco_captions: str = "./data/USCOCO_captions.json"
    uscoco_instances: str = "./data/USCOCO_instances.json"

    order_labels_by: str = "big_to_small"  # or 'left_to_right'

    filter_out_crowds: bool = True

    nb_of_pos_bins: int = 15
    pos_cont_pad_id: float = -1
    nobj_id: int = 200

    scale_lr_by_ddp_nodes: bool = False
    optimizer: str = "adamw"
    lr_schedule: Optional[str] = None  # 'linear_with_warmup'  # or None
    lr_schedule_warmup_epochs: int = 0

    latent_tree_h5_mask_value = -1

    syntax_train_json: str = "./data/train2017_spans.json"
    syntax_val_json: str = "./data/val2017_spans.json"
    syntax_absurd_json: str = "./data/USCOCO_spans.json"
    vocab_file: str = "./data/vpcfg_coco.dict.pkl"

    use_plm: bool = False
    use_tg: bool = False

    ann_trees_train_json: str = "./data/train2017_trees_with_tags.json"
    ann_trees_val_json: str = "./data/val2017_trees_with_tags.json"
    ann_trees_absurd_json: str = "./data/USCOCO_trees_with_tags.json"

    num_allowed_objects_in_image: int = -1
    new_valset_ids: str = "./data/train2017_new_valset_ids.json"
    use_new_valset: bool = True

    shuffle_sentences: bool = False
    shuffle_sentences_eval: bool = False

    use_cpn: bool = True
    cpn_extra_pad: float = 0.04
    old_preprocessing: bool = False
    old_preprocessing_clamp: bool = False
    old_coco_label_convert: bool = False
    old_preprocessing_skip_zero: bool = False
    old_prop_loss_nan: bool = False
    transform_min_visibility: float = 0.1

    use_smart_filter: bool = True
    nb_smart_filters: int = 2
    # load_smart_filter_from_file: bool = False
    smart_filter_compound_combination_type = "or"  # and | or
    # ('./data/sf_1.pkl', './data/sf_2.pkl')  # (None, None)
    smart_filter_pretrained_file: tuple = ("./data/sf_1.pkl", "./data/sf_2.pkl")
    # ('./data/sf_1.pkl', './data/sf_2.pkl')
    smart_filter_save_file: tuple = ("./data/sf_1.pkl", "./data/sf_2.pkl")
    smart_filter_normalize_dist: tuple = (True, True)
    smart_filter_distr_type: tuple = ("avgmax", "avgmax")  # avg | avgmax
    smart_filter_dist_type: tuple = ("reltomax", "abs")  # reltomax | abs
    smart_filter_discrimination_type: tuple = ("rel", "rel")  # rel | sd
    smart_filter_min_relative_size: tuple = (0.5, 0.5)
    smart_filter_max_deviation: tuple = (2.0, 2.0)

    use_image_transforms: bool = True
    p_HorizontalFlip: float = 0.5
    p_RandomSizedBBoxSafeCrop: float = 0.0
    width_RandomSizedBBoxSafeCrop: int = 256
    height_RandomSizedBBoxSafeCrop: int = 256
    p_RandomScale: float = 0.0
    max_RandomScale: float = 0.0

    continue_training_from_checkpoint: bool = False
    load_weights_from_checkpoint: bool = False
    checkpoint: str = ""
    load_avg_params_from: str = ""
    load_gen_state_dict_from: str = ""
    load_discs_state_dict_from: str = ""
    avg_params: bool = True
    cfg_name: str = ""

    debug: bool = False
    debug_max_epochs: int = 1
    profiler: Optional[str] = None
    overfit_on_val_samples: int = -1
    overfit_times_in_epoch: int = 10
    deterministic: bool = False
    wandb_offline: bool = False

    output_dir: str = "./output"
    run_output_dir: str = ""
    cuda: bool = True
    pin_memory: bool = True
    num_workers: int = 12
    persistent_workers: bool = False
    debug_num_workers: int = 4
    seed: int = 42
    do_train: bool = True
    do_test: bool = False
    do_validate: bool = False
    do_validate_during_training: bool = True
    batch_size: int = 128
    val_batch_size: int = 128
    num_words: int = 40

    optimize_data_loading: bool = True
    empty_cache: bool = False

    early_stop: str = "f1_iou_05"
    early_stop_min_or_max: str = "max"
    early_stop_patience: int = 20
    model_checkpoint_monitor_min_or_max: str = "max"
    model_checkpoint_monitor: str = "f1_iou_05"

    sample_loss_weights: bool = False

    total_parameters: int = 0
    trainable_parameters: int = 0

    flip_boxes: bool = True

    def configure(self) -> None:
        self.add_argument("--text_encoder", type=lambda x: TextEncoderConfig, required=False)
        self.add_argument("--probe", type=lambda x: ProbeConfig, required=False)
        self.add_argument("--lt", type=lambda x: TreeConfig, required=False)
        self.add_argument("--model", type=lambda x: ModelConfig, required=False)
        self.add_argument("--train", type=lambda x: TrainConfig, required=False)
        self.add_argument("--detr", type=lambda x: DETRConfig, required=False)
        self.add_argument("--obj_gan", type=lambda x: ObjGANConfig, required=False)

    def process_args(self, process=True) -> None:
        text_enc_cfg = TextEncoderConfig()
        text_enc_cfg.from_dict(self.text_encoder if hasattr(self, "text_encoder") else {})
        self.text_encoder: TextEncoderConfig = text_enc_cfg

        train_cfg = TrainConfig()
        train_cfg.from_dict(self.train if hasattr(self, "train") else {})
        self.train: TrainConfig = train_cfg

        lt_cfg = TreeConfig()
        lt_cfg.from_dict(self.lt if hasattr(self, "lt") else {})
        self.lt: TreeConfig = lt_cfg

        probe_cfg = ProbeConfig()
        probe_cfg.from_dict(self.probe if hasattr(self, "probe") else {})
        self.probe: ProbeConfig = probe_cfg

        detr_cfg = DETRConfig()
        detr_cfg.from_dict(self.detr if hasattr(self, "detr") else {})
        self.detr: DETRConfig = detr_cfg

        obj_gan = ObjGANConfig()
        obj_gan.from_dict(self.obj_gan if hasattr(self, "obj_gan") else {})
        self.obj_gan: ObjGANConfig = obj_gan

        model_cfg = ModelConfig()
        model_cfg.from_dict(self.model if hasattr(self, "model") else {})
        self.model: ModelConfig = model_cfg

        if process:
            self.process()

    def process(self):
        if not torch.cuda.is_available():
            self.train.gpus = None
            self.train.accelerator = "cpu"

        if self.model.obj_gan:
            self.model.autoregressive = True

        # self.text_encoder.max_target_positions = self.model.max_target_positions
        self.detr.max_num_generated_objects = self.model.max_target_positions
        # if self.mode == 'generate' and self.model.predict_num_queries:
        self.detr.num_queries = self.model.max_target_positions

        self.model.train_text_encoder = (
            self.text_encoder.txt_enc_finetune
        )  # or not self.text_encoder.txt_enc_pretrained
        self.model.predict_num_queries = (
            self.model.predict_num_queries and not self.model.autoregressive
        )

        if self.use_plm:
            self.text_encoder.text_encoder = "plm"
            self.text_encoder.hf_model_name_or_path = "gpt2"
            self.text_encoder.add_len_token = False
            if self.text_encoder.plm_add_structured_mask:
                assert "mask" in self.text_encoder.plm_checkpoint
        if self.use_tg:
            self.text_encoder.text_encoder = "tg"
            self.text_encoder.hf_model_name_or_path = "gpt2"
            self.text_encoder.add_len_token = False

        if not self.cuda:
            self.train.gpus = None

        self.text_encoder.add_len_token = (
            self.text_encoder.add_len_token
            and self.text_encoder.text_encoder in ("huggingface", "vokenization")
        )
        self.text_encoder.num_words = self.num_words

        if not self.model.autoregressive:
            self.detr.generation_strategy_for_eval_criterion = "detr"

        self.detr.load_span_tree_pos_embs = (
            self.detr.struct_loss_coef > 0.0 and self.detr.struct_loss_input == "tree_pos_embs"
        )

        if self.debug:
            logger.info("Disabling smart filters in debug mode")
            self.use_smart_filter = False

        if self.sample_loss_weights:
            self.model.label_loss_coef = random.choice([0.1, 0.25, 0.5, 1])
            self.model.struct_loss_coef = random.choice([0.1, 0.25, 0.5, 1])
            self.model.length_loss_coef = random.choice([0.1, 0.25, 0.5, 1])
            self.detr.bbox_loss_coef = random.choice([0.1, 0.5, 1, 5, 10])
            self.detr.giou_loss_coef = random.choice([0.1, 0.5, 1, 5, 10])
            self.detr.bbox_prop_loss_coef = random.choice([0.1, 0.5, 1, 5, 10])
            self.detr.bbox_rel_loss_coef = random.choice([0.1, 0.5, 1, 5, 10])

    def __getstate__(self):
        return self.to_dict()

    def to_dict(self):
        dct = self.as_dict()
        dct.update({"text_encoder": self.text_encoder.as_dict()})
        dct.update({"probe": self.probe.as_dict()})
        dct.update({"lt": self.lt.as_dict()})
        dct.update({"train": self.train.as_dict()})
        dct.update({"detr": self.detr.as_dict()})
        dct.update({"obj_gan": self.obj_gan.as_dict()})
        dct.update({"model": self.model.as_dict()})
        return {
            k: v if not isinstance(v, List) else str(v)
            for (k, v) in dct.items()
            if not isinstance(v, MethodType)
        }

    def __str__(self):
        return str(self.to_dict())


class ConfigNs(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
