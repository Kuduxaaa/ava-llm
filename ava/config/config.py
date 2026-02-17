class AvaConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=16,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        model_type="ava",
        # Mamba SSM parameters
        architecture_type="transformer",
        d_state=16,
        d_conv=4,
        expand=2,
        num_attention_layers=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.model_type = model_type

        # Mamba SSM parameters
        self.architecture_type = architecture_type
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_attention_layers = num_attention_layers

        self.predefined_models = [
            "100m",
            "500m",
            "1b",
            "3b",
            "7b",
            "13b",
            "30b",
            "65b",
            "100b",
            "mamba-130m",
            "hybrid-130m",
        ]

        if "head_dim" in kwargs:
            self.head_dim = kwargs["head_dim"]
        else:
            self.head_dim = hidden_size // num_attention_heads

        if "kv_heads" in kwargs:
            self.kv_heads = kwargs["kv_heads"]
        else:
            self.kv_heads = num_attention_heads

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def apply_for(self, model="7b"):
        if model not in self.predefined_models:
            raise ValueError(f'Configuration for "{model}" is not defined.')

        # Tiny models (Edge devices, IoT, offline agents, chatbots)
        if model == "100m":
            self.hidden_size = 768
            self.intermediate_size = 3072
            self.num_hidden_layers = 6
            self.num_attention_heads = 12
            self.max_position_embeddings = 2048
            self.head_dim = 64
            self.kv_heads = 4

        elif model == "500m":
            self.hidden_size = 1024
            self.intermediate_size = 4096
            self.num_hidden_layers = 8
            self.num_attention_heads = 16
            self.max_position_embeddings = 2048
            self.head_dim = 64
            self.kv_heads = 4

        # Small models (Mobile apps, personal assistants, summarization)
        elif model == "1b":
            self.hidden_size = 1280
            self.intermediate_size = 5120
            self.num_hidden_layers = 12
            self.num_attention_heads = 16
            self.max_position_embeddings = 4096
            self.head_dim = 80
            self.kv_heads = 8

        elif model == "3b":
            self.hidden_size = 1600
            self.intermediate_size = 6400
            self.num_hidden_layers = 24
            self.num_attention_heads = 16
            self.max_position_embeddings = 4096
            self.head_dim = 100
            self.kv_heads = 8

        # Medium models (Coding, reasoning, multi-turn chat, translation)
        elif model == "7b":
            self.hidden_size = 4096
            self.intermediate_size = 11008
            self.num_hidden_layers = 32
            self.num_attention_heads = 32
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        elif model == "13b":
            self.hidden_size = 5120
            self.intermediate_size = 13824
            self.num_hidden_layers = 40
            self.num_attention_heads = 40
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        # Large models (Research, enterprise-level applications)
        elif model == "30b":
            self.hidden_size = 6656
            self.intermediate_size = 17920
            self.num_hidden_layers = 60
            self.num_attention_heads = 52
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        elif model == "65b":
            self.hidden_size = 8192
            self.intermediate_size = 22016
            self.num_hidden_layers = 80
            self.num_attention_heads = 64
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        # Massive models (AGI research, cutting-edge LLMs)
        elif model == "100b":
            self.hidden_size = 12288
            self.intermediate_size = 33024
            self.num_hidden_layers = 96
            self.num_attention_heads = 96
            self.max_position_embeddings = 16384
            self.head_dim = 128
            self.kv_heads = 8

        # Mamba SSM models
        elif model == "mamba-130m":
            self.architecture_type = "mamba"
            self.hidden_size = 768
            self.intermediate_size = 3072
            self.num_hidden_layers = 12
            self.num_attention_heads = 12
            self.max_position_embeddings = 2048
            self.head_dim = 64
            self.kv_heads = 4
            self.d_state = 16
            self.d_conv = 4
            self.expand = 2
            self.num_attention_layers = 0

        elif model == "hybrid-130m":
            self.architecture_type = "hybrid"
            self.hidden_size = 768
            self.intermediate_size = 3072
            self.num_hidden_layers = 12
            self.num_attention_heads = 12
            self.max_position_embeddings = 2048
            self.head_dim = 64
            self.kv_heads = 4
            self.d_state = 16
            self.d_conv = 4
            self.expand = 2
            self.num_attention_layers = 2

        return self
