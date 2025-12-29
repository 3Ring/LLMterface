from enum import Enum


class GenericModelType(Enum):
    text_lite = "generic-lite"
    text_standard = "generic-standard"
    text_heavy = "generic-heavy"
    embedding = "generic-embedding"
    image_gen_lite = "generic-image-gen-lite"
    image_gen_standard = "generic-image-gen-standard"
    image_gen_heavy = "generic-image-gen-heavy"
    audio_gen_lite = "generic-audio-gen-lite"
    audio_gen_standard = "generic-audio-gen-standard"
    audio_gen_heavy = "generic-audio-gen-heavy"
    video_gen_lite = "generic-video-gen-lite"
    video_gen_standard = "generic-video-gen-standard"
    video_gen_heavy = "generic-video-gen-heavy"
