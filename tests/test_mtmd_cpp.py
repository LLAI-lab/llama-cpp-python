import ctypes
import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _init_opt_available(module) -> bool:
    return not getattr(module.mtmd_helper_init_opt_default, "__ctypes_optional__", False)


def test_import_mtmd_cpp():
    module = importlib.import_module("llama_cpp.mtmd_cpp")

    assert module is not None


def test_mtmd_helper_init_opt_abi():
    module = importlib.import_module("llama_cpp.mtmd_cpp")

    assert module.mtmd_helper_video_init_params._fields_ == [
        ("fps_target", ctypes.c_float),
        ("ffmpeg_bin_dir", ctypes.c_char_p),
        ("timestamp_interval_ms", ctypes.c_int64),
    ]
    assert module.mtmd_helper_init_opt._fields_ == [
        ("video_params", module.mtmd_helper_video_init_params),
    ]
    assert module.mtmd_helper_bitmap_init_from_file.argtypes == [
        module.mtmd_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_bool,
        module.mtmd_helper_init_opt,
    ]
    assert module.mtmd_helper_bitmap_init_from_buf.argtypes == [
        module.mtmd_context_p_ctypes,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_bool,
        module.mtmd_helper_init_opt,
    ]
    assert module.mtmd_helper_video_init.restype is module.mtmd_helper_video_p_ctypes

    if not _init_opt_available(module):
        # kv-stream fork builds do not export mtmd_helper_init_opt_default
        pytest.skip("mtmd_helper_init_opt_default is unavailable in this build")
    opt = module.mtmd_helper_init_opt_default()
    assert opt.video_params.fps_target == 4.0
    assert opt.video_params.ffmpeg_bin_dir is None
    assert opt.video_params.timestamp_interval_ms == 5000


@pytest.mark.parametrize("handler_name", ["MTMDBaseHandler", "MTMDChatHandler"])
def test_mtmd_chat_handler_video_options(tmp_path, handler_name):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler_class = getattr(multimodal, handler_name)

    executable_suffix = ".exe" if os.name == "nt" else ""
    for executable_name in ("ffmpeg", "ffprobe"):
        executable_path = tmp_path / (executable_name + executable_suffix)
        executable_path.touch()
        executable_path.chmod(0o755)

    handler = handler_class(
        mmproj_path=str(tmp_path),
        video_fps_target=2.5,
        video_ffmpeg_bin_dir=tmp_path,
        video_timestamp_interval_ms=10000,
    )
    video_params = handler._mtmd_helper_init_opt.video_params
    assert video_params.fps_target == 2.5
    assert video_params.ffmpeg_bin_dir == os.fsencode(os.path.abspath(tmp_path))
    assert video_params.timestamp_interval_ms == 10000

    if _init_opt_available(module):
        opt = module.mtmd_helper_init_opt_default()
    else:
        # kv-stream fork builds: zero-initialized struct is the runtime fallback
        opt = module.mtmd_helper_init_opt()
    handler = handler_class(
        mmproj_path=str(tmp_path),
        mtmd_helper_init_opt=opt,
    )
    assert handler._mtmd_helper_init_opt is opt

    with pytest.raises(ValueError, match="cannot be combined"):
        handler_class(
            mmproj_path=str(tmp_path),
            mtmd_helper_init_opt=opt,
            video_fps_target=1.0,
        )


def test_mtmd_chat_handler_rejects_invalid_ffmpeg_bin_dir(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")

    with pytest.raises(ValueError, match="is not an existing directory"):
        multimodal.MTMDChatHandler(
            mmproj_path=str(tmp_path),
            video_ffmpeg_bin_dir=tmp_path / "missing",
        )

    with pytest.raises(ValueError, match="ffmpeg and ffprobe"):
        multimodal.MTMDChatHandler(
            mmproj_path=str(tmp_path),
            video_ffmpeg_bin_dir=tmp_path,
        )


@pytest.mark.parametrize("handler_name", ["MTMDBaseHandler", "MTMDChatHandler"])
def test_mtmd_context_initialization_and_close(tmp_path, handler_name):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = getattr(multimodal, handler_name)(
        clip_model_path=str(tmp_path), verbose=False, use_gpu=False,
        image_min_tokens=32, image_max_tokens=128, batch_max_tokens=256,
    )
    native = handler._mtmd_cpp
    backend = SimpleNamespace(
        mtmd_context_params_default=native.mtmd_context_params_default,
        clip_flash_attn_type=native.clip_flash_attn_type,
        mtmd_helper_log_set=Mock(),
        mtmd_default_marker=Mock(return_value=b"<media>"),
        mtmd_init_from_file=Mock(return_value=123),
        mtmd_support_vision=Mock(return_value=True),
        mtmd_support_audio=Mock(return_value=False),
        mtmd_helper_support_video=Mock(return_value=True),
        mtmd_free=Mock(),
    )
    handler._mtmd_cpp = backend
    model = SimpleNamespace(model=456, n_threads=2)
    if handler_name == "MTMDChatHandler":
        model.token_eos = lambda: 2
        model.token_bos = lambda: 1
        model.detokenize = lambda tokens: {1: b"<bos>", 2: b"<eos>"}[tokens[0]]
    try:
        handler._init_mtmd_context(model)
        handler._init_mtmd_context(model)
        backend.mtmd_init_from_file.assert_called_once()
        assert handler.media_marker == "<media>"
        assert handler.is_support_vision
        assert not handler.is_support_audio
        assert handler.is_support_video
        assert handler.mctx_params.n_threads == 2
        assert not handler.mctx_params.use_gpu
        assert handler.mctx_params.image_min_tokens == 32
        assert handler.mctx_params.image_max_tokens == 128
        assert handler.mctx_params.batch_max_tokens == 256
        if handler_name == "MTMDChatHandler":
            assert handler.mtmd_bos_token == "<bos>"
            assert handler.mtmd_eos_token == "<eos>"
        else:
            assert not hasattr(handler, "chat_template")
            assert not hasattr(handler, "mtmd_bos_token")
    finally:
        handler.close()
        handler.close()
    backend.mtmd_free.assert_called_once_with(123)


def test_mtmd_base_close_releases_context_after_callback_failure(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = multimodal.MTMDBaseHandler(mmproj_path=str(tmp_path), verbose=False)
    events = []
    handler.mtmd_ctx = 123
    handler._mtmd_cpp = SimpleNamespace(mtmd_free=lambda ctx: events.append("context"))

    def release_child():
        events.append("child")
        raise RuntimeError("child cleanup failed")

    handler._exit_stack.callback(release_child)
    with pytest.raises(RuntimeError, match="child cleanup failed"):
        handler.close()
    handler.close()
    assert events == ["child", "context"]
    assert handler.mtmd_ctx is None


def test_mtmd_chat_constructor_preserves_template_options(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")

    class CustomChatHandler(multimodal.MTMDChatHandler):
        chat_format = "custom: {{ value }}"

    arguments = {"value": "hello"}
    handler = CustomChatHandler(
        str(tmp_path), False, True, -1, -1, "override", 512, arguments,
    )
    try:
        assert isinstance(handler, multimodal.MTMDBaseHandler)
        assert handler.chat_template.render(**handler.extra_template_arguments) == "custom: hello"
        arguments["value"] = "changed"
        assert handler.extra_template_arguments == {"value": "hello"}
    finally:
        handler.close()


def test_mtmd_base_image_loader_uses_subclass_byte_loader():
    import io
    from PIL import Image
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    payload = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 128)).save(payload, format="PNG")

    class CustomMediaHandler(multimodal.MTMDBaseHandler):
        @staticmethod
        def _load_bytes(media_url, timeout=15, kind="media"):
            assert media_url == "custom-image"
            assert kind == "image"
            return payload.getvalue()

    result = CustomMediaHandler._load_image("custom-image")
    with Image.open(io.BytesIO(result)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (2, 2)
