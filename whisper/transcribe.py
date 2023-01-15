import argparse
import os
import warnings
import copy
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from itertools import repeat

import numpy as np
import torch
import tqdm

from .audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram, load_audio_waveform_img, remove_lower_quantile, wave_to_ts_filter
from .decoding import DecodingOptions, DecodingResult
from .stabilization import stabilize_timestamps, add_whole_word_ts
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt

if TYPE_CHECKING:
    from .model import Whisper


# modified version of whisper.transcribe.transcribe
def transcribe(
        model: "Whisper",
        audio: Union[str, List, np.ndarray, torch.Tensor],
        *,
        verbose: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        language_threshold: Optional[float] = 0.6,
        language_detection_segments: int = 1,
        condition_on_previous_text: bool = True,
        stab=True, top_focus=False, ts_num: int = 10,
        alpha: float = None, print_unstab=False, pbar=False,
        suppress_silence: bool = True,
        suppress_middle: bool = True,
        suppress_word_ts: bool = True,
        remove_background: bool = True,
        silence_threshold: float = 0.1,
        prepend_punctuations: Union[List[str], Tuple[str]] = None,
        append_punctuations: Union[List[str], Tuple[str]] = None,
        audio_for_mask: (str, bytes) = None,
        **decode_options):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the decoded text (with finalized timestamps) to the console (Default: False)
        Use print_unstab for previous behavior of verbose but with token timestamps

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    stab: bool
        Stabilizing timestamps by cross compare timestamps and using additional top timestamp predictions
        to fill in when appropriate to ensure timestamps are chronological.

    top_focus: bool
        Adhere closely to the top predictions for token timestamps stabilization

    ts_num: int
        Number of top timestamp predictions to save for each word for postprocessing stabilization (default: 10).

    alpha: float
        Amount of noise to add to audio to produce slightly difference results.
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    print_unstab: bool
        Whether to display the text (without stabilize timestamps) being decoded to the console (Default: False)
        (i.e. behaves like verbose before model was modified and progress bar will be disabled if True)

    pbar: bool
        Whether to enable progress bar for the decoding process (Default: False). Ignored if print_unstab=True

    suppress_silence: bool
        Suppress timestamp tokens that are marked as silent

    suppress_middle: bool
        Suppress any silent timestamps tokens of middle of the segment instead of only beginning and ending

    suppress_word_ts: bool
        Suppress timestamp tokens of words that are marked as silent

    remove_background: bool
        Whether to remove background noise from waveform so that it is marked silent.
        Determined by parameters part of decode_options (i.e. specify like other options here):
            upper_quantile: float
                The upper quantile of amplitude to determine a max amplitude, mx (Default: 0.85)
            lower_quantile: float
                The lower quantile of amplitude to determine a min amplitude, mn (Default: 0.15)
            lower_threshold: float
                Suppressed sections of waveform where amplitude < lower_threshold*(mx-mn) + mn. (Default: 0.15)

    silence_threshold: float:
        Audio segments silence average >= silence_threshold
        then that segment will not have background removed even if remove_background=True.
        e.g. 0.5 means if less than half of the audio segment is silent then background will be removed accordingly

    prepend_punctuations: Union[List[str], Tuple[str]]
        Punctuations to prepend to next word (Default: “¿([{)

    append_punctuations: Union[List[str], Tuple[str]]
        Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)

    audio_for_mask: (str, bytes)
        Original audio track as path or bytes of audio file.
        Since resampled audio may shift the waveform image,
        this is an alternative to 'audio' option to generate suppression mask from the original audio.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    if 'no_captions_threshold' in decode_options:
        warnings.warn('no_captions_threshold is deprecated. '
                      'Please use no_speech_threshold instead.', DeprecationWarning, stacklevel=2)
        no_speech_threshold = decode_options.pop('no_captions_threshold')

    if type(audio) == list:
        return batch_transcribe(model=model,
                                audio=audio,
                                verbose=verbose,
                                temperature=temperature,
                                compression_ratio_threshold=compression_ratio_threshold,
                                logprob_threshold=logprob_threshold,
                                no_speech_threshold=no_speech_threshold,
                                condition_on_previous_text=condition_on_previous_text,
                                stab=stab, top_focus=top_focus, ts_num=ts_num,
                                alpha=alpha, print_unstab=print_unstab, pbar=pbar,
                                suppress_silence=suppress_silence,
                                suppress_middle=suppress_middle,
                                suppress_word_ts=suppress_word_ts,
                                remove_background=remove_background,
                                silence_threshold=silence_threshold,
                                prepend_punctuations=prepend_punctuations,
                                append_punctuations=append_punctuations,
                                audio_for_mask=audio_for_mask,
                                **decode_options)

    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if 'max_initial_timestamp' not in decode_options:
        decode_options['max_initial_timestamp'] = None

    mel = log_mel_spectrogram(audio)
    num_frames = mel.shape[-1]

    if decode_options.get("language", None) is None:
        if verbose:
            print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        if language_detection_segments is None or language_detection_segments < 1:
            language_detection_segments = 1
        seek = 0
        languages = []
        while seek < num_frames and seek < N_FRAMES * language_detection_segments:
            segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(segment)
            lang = max(probs, key=probs.get)
            lang_prob = probs[lang]
            if language_threshold is not None and lang_prob > language_threshold:
                decode_options["language"] = lang
                break
            else:
                languages.append(lang)
                seek += segment.shape[-1]
        else:
            # If no language detected for all segments, the majority vote of the highest projected
            # languages for all segments is used to determine the language.
            decode_options["language"] = max(set(languages), key=languages.count)

    mel = mel.unsqueeze(0)
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    ignore_shift = decode_options.pop('ignore_shift', False)

    def decode_with_fallback(segment: torch.Tensor, suppress_ts_mask: Tensor = None) \
            -> Union[List[DecodingResult], tuple]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        kwargs = {**decode_options}
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results, ts_tokens, ts_logits_, tc = model.decode(segment, options, ts_num=ts_num, alpha=alpha,
                                                      suppress_ts_mask=suppress_ts_mask,
                                                      suppress_word_ts=suppress_word_ts)  # my

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            # TODO This part needs to be adapted into batch inference
            needs_fallback = [
                compression_ratio_threshold is not None
                and result.compression_ratio > compression_ratio_threshold
                or logprob_threshold is not None
                and result.avg_logprob < logprob_threshold
                for result in results
            ]

            if any(needs_fallback):
                options = DecodingOptions(**kwargs, temperature=t)
                retries, r_ts_tokens, r_ts_logits, tc = model.decode(segment[needs_fallback], options,
                                                                 ts_num=ts_num, alpha=alpha,
                                                                 suppress_ts_mask=suppress_ts_mask,
                                                                 suppress_word_ts=suppress_word_ts)
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]
                    ts_tokens[original_index] = r_ts_tokens[retry_index]
                    ts_logits_[original_index] = r_ts_logits[retry_index]

        return results, ts_tokens, ts_logits_, tc

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def _to_list(x: (Tensor, None)):
        if x is None:
            return x
        return x.tolist()

    def add_segment(
            *, offset: float, start: float, end: float, text_tokens: Tensor, result: DecodingResult,
            start_timestamps: list = None, end_timestamps: list = None, word_timestamps: Tensor = None,
            start_ts_logits: list = None, end_ts_logits: list = None, word_ts_logits: Tensor = None,
            tc_logits: Tensor = None
    ):
        no_eot_mask = text_tokens < tokenizer.eot
        text_tokens_no_eot = text_tokens[no_eot_mask]
        text = tokenizer.decode(text_tokens_no_eot)

        if len(text.strip()) == 0:  # skip empty text output
            return

        if word_timestamps is not None:
            assert word_timestamps.shape[0] == text_tokens.shape[0]
            if word_ts_logits is None:
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], repeat(None))
            else:
                assert word_ts_logits.shape[0] == text_tokens.shape[0]
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], word_ts_logits[no_eot_mask])

            word_timestamps = [dict(word=tokenizer.decode([token]),
                                    token=token.item(),
                                    timestamps=timestamps_.tolist(),
                                    timestamp_logits=_to_list(ts_logits_))
                               for token, timestamps_, ts_logits_ in word_ts_fields]

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                'offset': offset,  # offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": _get_new_attrs(result, 'no_caption_prob'),
                "alt_start_timestamps": start_timestamps,
                "start_ts_logits": start_ts_logits,
                "alt_end_timestamps": end_timestamps,
                "end_ts_logits": end_ts_logits,
                "unstable_word_timestamps": word_timestamps,
                'anchor_point': False,
                "confidence_score": tc_logits  # my
            }
        )
        if print_unstab or (verbose and not stab):
            print(f'[{format_timestamp(start)} --> {format_timestamp(end)}] "{text}"')
            if word_timestamps is not None:
                ts_str = (f' ->[{format_timestamp(ts_["timestamps"][0])}] "{ts_["word"].strip()}"' for ts_ in
                          word_timestamps)
                print('\n'.join(ts_str), end='\n\n')

    if suppress_silence:
        all_silent = False
        ts_scale = HOP_LENGTH / SAMPLE_RATE / time_precision
        wfh, wfw = 100, int(mel.shape[-1] * ts_scale)
        wf = load_audio_waveform_img(audio_for_mask or audio, wfh, wfw, ignore_shift=ignore_shift)
        if not wf.any():
            if audio_for_mask:
                wf = load_audio_waveform_img(load_audio(audio) if isinstance(audio, str) else audio,
                                             wfh, wfw, ignore_shift=True)
            else:
                if isinstance(audio, str):
                    wf = load_audio_waveform_img(load_audio(audio), wfh, wfw, ignore_shift=True)
                else:
                    all_silent = True

            if not all_silent:
                all_silent = not wf.any()
            if all_silent:
                warnings.warn('The audio appears to be entirely silent. suppress_silence will be set to False',
                              stacklevel=2)
                suppress_silence = False

    upper_quantile = decode_options.pop('upper_quantile', 0.85)
    lower_quantile = decode_options.pop('lower_quantile', 0.15)
    lower_threshold = decode_options.pop('lower_threshold', 0.15)

    with tqdm(total=num_frames, unit='frames', disable=(print_unstab or not pbar)) as tqdm_pbar:

        def update_pbar():
            if not tqdm_pbar.disable:
                tqdm_pbar.update(min(num_frames, seek) - tqdm_pbar.n)

        while seek < mel.shape[-1]:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            remaining_duration = float((mel.shape[-1] - seek) * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES).to(model.device).to(dtype)
            segment_duration = min(float(segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE), remaining_duration)
            segment_max_ts = segment_duration / time_precision

            if suppress_silence:
                wf_seek = int(seek * ts_scale)
                segment_wf = wf[..., wf_seek:wf_seek + 1501]
                if remove_background and \
                        (1 - segment_wf.sum(0).clip(max=1).mean()) < silence_threshold:
                    segment_wf = remove_lower_quantile(segment_wf.astype(np.float32),
                                                       upper_quantile=upper_quantile,
                                                       lower_quantile=lower_quantile,
                                                       lower_threshold=lower_threshold)
                segment_wf = pad_or_trim(segment_wf, 1501)
                suppress_ts_mask = torch.from_numpy(wave_to_ts_filter(segment_wf,
                                                                      suppress_middle=suppress_middle,
                                                                      max_index=int(segment_max_ts)))

                if suppress_ts_mask.all():  # segment is silent
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    update_pbar()
                    continue
            else:
                suppress_ts_mask = None

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result, finalized_ts_tokens, ts_logits, tc = decode_with_fallback(segment,
                                                                          suppress_ts_mask=suppress_ts_mask)

            result = result[0]
            tokens = torch.tensor(result.tokens)
            finalized_ts_tokens = torch.tensor(finalized_ts_tokens[0])
            ts_logits = torch.tensor(ts_logits[0])

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = _get_new_attrs(result, 'no_caption_prob') > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    sliced_ts_tokens = finalized_ts_tokens[last_slice:current_slice]
                    sliced_ts_logits = ts_logits[last_slice:current_slice]
                    sliced_tc = tc[last_slice:current_slice]
                    start_timestamp_position = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                            sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )

                    word_ts = timestamp_offset + sliced_ts_tokens * time_precision

                    add_segment(
                        offset=timestamp_offset,
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=min(timestamp_offset + end_timestamp_position * time_precision,
                                timestamp_offset + segment_duration),
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                        start_timestamps=word_ts[0].tolist(),
                        end_timestamps=word_ts[-1].tolist(),
                        word_timestamps=word_ts[1:-1],
                        start_ts_logits=sliced_ts_logits[0].tolist(),
                        end_ts_logits=sliced_ts_logits[-1].tolist(),
                        word_ts_logits=sliced_ts_logits[1:-1],
                        tc_logits=sliced_tc[1:-1]  # my
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                    min(tokens[last_slice - 1].item() - tokenizer.timestamp_begin, segment_max_ts)
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = min(timestamps[-1].item() - tokenizer.timestamp_begin, segment_max_ts)
                    duration = last_timestamp_position * time_precision

                word_ts = timestamp_offset + finalized_ts_tokens * time_precision

                add_segment(
                    offset=timestamp_offset,
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                    word_timestamps=word_ts,
                    word_ts_logits=ts_logits,
                    tc_logits=tc
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if all_segments:
                all_segments[-1]['anchor_point'] = True
                all_segments[-1]['next_offset'] = float(seek * HOP_LENGTH / SAMPLE_RATE)
            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            update_pbar()

    if len(all_segments) > 1 and all_segments[-1]['alt_start_timestamps'] is None:
        all_segments[-1]['alt_start_timestamps'] = all_segments[-2]['alt_end_timestamps']

    # # my prepare confidence
    # all_segments = my_prepare_confidence_and_words(tokenizer, all_segments)  # side effect

    if stab:
        all_segments = stabilize_timestamps(all_segments, top_focus=top_focus)
        add_whole_word_ts(tokenizer, all_segments,
                          merge_non_space=True,  # my
                          prepend_punctuations=prepend_punctuations,
                          append_punctuations=append_punctuations)
        if verbose:
            print('\nSTABILIZED:')
            for seg_ in all_segments:
                print(f'[{format_timestamp(seg_["start"])} --> {format_timestamp(seg_["end"])}] "{seg_["text"]}"')
                if seg_['word_timestamps']:
                    ts_str = (f' ->[{format_timestamp(ts_["timestamp"])}] "{ts_["word"].strip()}"' for ts_ in
                               seg_['word_timestamps'])
                    print('\n'.join(ts_str), end='\n\n')
        import pickle
        with open('filename.pickle', 'wb') as fh:
            pickle.dump(all_segments, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)

# modified version of whisper.transcribe.transcribe to allow for batch inference
def batch_transcribe(
        model: "Whisper",
        audio: Union[str, List, np.ndarray, torch.Tensor],
        *,
        verbose: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        stab=True, top_focus=False, ts_num: int = 10,
        alpha: float = None, print_unstab=False, pbar=False,
        suppress_silence: bool = True,
        suppress_middle: bool = True,
        suppress_word_ts: bool = True,
        remove_background: bool = True,
        silence_threshold: float = 0.1,
        prepend_punctuations: Union[List[str], Tuple[str]] = None,
        append_punctuations: Union[List[str], Tuple[str]] = None,
        audio_for_mask: (str, bytes) = None,
        **decode_options):
    """
    Transcribe multiple audio files in parallel using the batch dimension of the Whisper model

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The list of paths to the audio files to open, or the audio waveforms

    verbose: bool
        Whether to display the decoded text (with finalized timestamps) to the console (Default: False)
        Use print_unstab for previous behavior of verbose but with token timestamps

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    stab: bool
        Stabilizing timestamps by cross compare timestamps and using additional top timestamp predictions
        to fill in when appropriate to ensure timestamps are chronological.

    top_focus: bool
        Adhere closely to the top predictions for token timestamps stabilization

    ts_num: int
        Number of top timestamp predictions to save for each word for postprocessing stabilization (default: 10).

    alpha: float
        Amount of noise to add to audio to produce slightly difference results.
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    print_unstab: bool
        Whether to display the text (without stabilize timestamps) being decoded to the console (Default: False)
        (i.e. behaves like verbose before model was modified and progress bar will be disabled if True)

    pbar: bool
        Whether to enable progress bar for the decoding process (Default: False). Ignored if print_unstab=True

    suppress_silence: bool
        Suppress timestamp tokens that are marked as silent

    suppress_middle: bool
        Suppress any silent timestamps tokens of middle of the segment instead of only beginning and ending

    suppress_word_ts: bool
        Suppress timestamp tokens of words that are marked as silent

    remove_background: bool
        Whether to remove background noise from waveform so that it is marked silent.
        Determined by parameters part of decode_options (i.e. specify like other options here):
            upper_quantile: float
                The upper quantile of amplitude to determine a max amplitude, mx (Default: 0.85)
            lower_quantile: float
                The lower quantile of amplitude to determine a min amplitude, mn (Default: 0.15)
            lower_threshold: float
                Suppressed sections of waveform where amplitude < lower_threshold*(mx-mn) + mn. (Default: 0.15)

    silence_threshold: float:
        Audio segments silence average >= silence_threshold
        then that segment will not have background removed even if remove_background=True.
        e.g. 0.5 means if less than half of the audio segment is silent then background will be removed accordingly

    prepend_punctuations: Union[List[str], Tuple[str]]
        Punctuations to prepend to next word (Default: “¿([{)

    append_punctuations: Union[List[str], Tuple[str]]
        Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)

    audio_for_mask: (str, bytes)
        Original audio track as path or bytes of audio file.
        Since resampled audio may shift the waveform image,
        this is an alternative to 'audio' option to generate suppression mask from the original audio.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A list of dictionaries containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    if 'no_captions_threshold' in decode_options:
        warnings.warn('no_captions_threshold is deprecated. '
                      'Please use no_speech_threshold instead.', DeprecationWarning, stacklevel=2)
        no_speech_threshold = decode_options.pop('no_captions_threshold')


    batch_size = len(audio)
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if 'max_initial_timestamp' not in decode_options:
        decode_options['max_initial_timestamp'] = None

    mels = [log_mel_spectrogram(audio_file) for audio_file in audio]
    segments = [pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype) for mel in mels]

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            languages = ['en'] * len(audio)
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            language_probs = [model.detect_language(segment)[1] for segment in segments]
            languages = [max(probs, key=probs.get) for probs in language_probs]
            if verbose is not None:
                print(f"Detected languages: {[LANGUAGES[opt].title() for opt in languages]}")
    else:
        lang = decode_options.get("language")
        if type(lang) == str:
            languages = [lang] * len(audio)
        elif type(lang) == list:
            assert all(isinstance(l, str) for l in
                       lang), "If a list of languages is specified in DecodeOptions, all languages must be strings."
            assert len(lang) == len(
                audio), "If a list of languages is specified in DecodeOptions, the list length must match the number of audio files specified."
            languages = lang
        else:
            raise NotImplementedError("Only string and list arguments are supported for the language DecodeOption.")

    mels = [mel.unsqueeze(0) for mel in mels]
    task = decode_options.get("task", "transcribe")
    tokenizers = {}
    for lang in languages:
        if lang not in tokenizers.keys():
            tokenizers[lang] = get_tokenizer(model.is_multilingual, language=lang, task=task)

    ignore_shift = decode_options.pop('ignore_shift', False)

    def decode_with_fallback(segment: torch.Tensor, suppress_ts_mask: Tensor = None) \
            -> Union[List[DecodingResult], tuple]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        results = None
        ts_tokens = None
        ts_logits_ = None
        tc = None
        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            results, ts_tokens, ts_logits_, tc = model.decode(segment, options, ts_num=ts_num, alpha=alpha,
                                                          suppress_ts_mask=suppress_ts_mask,
                                                          suppress_word_ts=suppress_word_ts)

            needs_fallback = False
            if type(results) == list:
                for result in results:
                    if compression_ratio_threshold is not None and result.compression_ratio > compression_ratio_threshold:
                        needs_fallback = True  # too repetitive
                    if logprob_threshold is not None and result.avg_logprob < logprob_threshold:
                        needs_fallback = True  # average log probability is too low
            else:
                if compression_ratio_threshold is not None and result.compression_ratio > compression_ratio_threshold:
                    needs_fallback = True  # too repetitive
                if logprob_threshold is not None and result.avg_logprob < logprob_threshold:
                    needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return results, ts_tokens, ts_logits_, tc

        # kwargs = {**decode_options}
        # t = temperatures[0]
        # if t == 0:
        #     best_of = kwargs.pop("best_of", None)
        # else:
        #     best_of = kwargs.get("best_of", None)
        #
        # options = DecodingOptions(**kwargs, temperature=t)
        # results, ts_tokens, ts_logits_, tc = model.decode(segment, options, ts_num=ts_num, alpha=alpha,
        #                                               suppress_ts_mask=suppress_ts_mask,
        #                                               suppress_word_ts=suppress_word_ts)
        #
        # kwargs.pop("beam_size", None)  # no beam search for t > 0
        # kwargs.pop("patience", None)  # no patience for t > 0
        # kwargs["best_of"] = best_of  # enable best_of for t > 0
        # for t in temperatures[1:]:
        #     needs_fallback = [
        #         compression_ratio_threshold is not None
        #         and result.compression_ratio > compression_ratio_threshold
        #         or logprob_threshold is not None
        #         and result.avg_logprob < logprob_threshold
        #         for result in results
        #     ]
        #     if any(needs_fallback):
        #         options = DecodingOptions(**kwargs, temperature=t)
        #         retries, r_ts_tokens, r_ts_logits, tc = model.decode(segment[needs_fallback], options,
        #                                                          ts_num=ts_num, alpha=alpha,
        #                                                          suppress_ts_mask=suppress_ts_mask,
        #                                                          suppress_word_ts=suppress_word_ts)
        #         for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
        #             results[original_index] = retries[retry_index]
        #             ts_tokens[original_index] = r_ts_tokens[retry_index]
        #             ts_logits_[original_index] = r_ts_logits[retry_index]
        #
        # return results, ts_tokens, ts_logits_, tc

    seekers = [0] * len(audio)
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = [[] for _ in range(batch_size)]
    all_segments = [[] for _ in range(batch_size)]
    prompt_reset_since = [0] * batch_size

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    initial_prompts = []
    if initial_prompt:
        assert len(initial_prompt) == batch_size, "Number of initial prompts must match batch size."
        for i in range(batch_size):
            initial_prompts.append(tokenizers[languages[i]].encode(" " + initial_prompt[i].strip()))
            all_tokens.extend(initial_prompt)

    def _to_list(x: (Tensor, None)):
        if x is None:
            return x
        return x.tolist()

    def add_segment(
            *, seeker: int, offset: float, start: float, end: float, text_tokens: Tensor, result: DecodingResult,
            tokenizer, segments, start_timestamps: list = None, end_timestamps: list = None, word_timestamps: Tensor = None,
            start_ts_logits: list = None, end_ts_logits: list = None, word_ts_logits: Tensor = None,
            tc_logits: Tensor = None
    ):
        no_eot_mask = text_tokens < tokenizer.eot
        text_tokens_no_eot = text_tokens[no_eot_mask]
        text = tokenizer.decode(text_tokens_no_eot)

        if len(text.strip()) == 0:  # skip empty text output
            return

        if word_timestamps is not None:
            assert word_timestamps.shape[0] == text_tokens.shape[0]
            if word_ts_logits is None:
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], repeat(None))
            else:
                assert word_ts_logits.shape[0] == text_tokens.shape[0]
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], word_ts_logits[no_eot_mask])

            word_timestamps = [dict(word=tokenizer.decode([token]),
                                    token=token.item(),
                                    timestamps=timestamps_.tolist(),
                                    timestamp_logits=_to_list(ts_logits_))
                               for token, timestamps_, ts_logits_ in word_ts_fields]

        segments.append(
            {
                "id": len(all_segments),
                "seek": seeker,
                'offset': offset,  # offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": _get_new_attrs(result, 'no_caption_prob'),
                "alt_start_timestamps": start_timestamps,
                "start_ts_logits": start_ts_logits,
                "alt_end_timestamps": end_timestamps,
                "end_ts_logits": end_ts_logits,
                "unstable_word_timestamps": word_timestamps,
                'anchor_point': False,
                "confidence_score": tc_logits  # my
            }
        )
        if print_unstab or (verbose and not stab):
            print(f'[{format_timestamp(start)} --> {format_timestamp(end)}] "{text}"')
            if word_timestamps is not None:
                ts_str = (f' ->[{format_timestamp(ts_["timestamps"][0])}] "{ts_["word"].strip()}"' for ts_ in
                          word_timestamps)
                print('\n'.join(ts_str), end='\n\n')

    batch_suppress_silence = [suppress_silence] * len(mels)
    for i in range(len(mels)):
        if batch_suppress_silence[i]:
            all_silent = False
            ts_scale = HOP_LENGTH / SAMPLE_RATE / time_precision
            wfh, wfw = 100, int(mels[i].shape[-1] * ts_scale)
            wf = load_audio_waveform_img(audio_for_mask or audio[i], wfh, wfw, ignore_shift=ignore_shift)
            if not wf.any():
                if audio_for_mask:
                    wf = load_audio_waveform_img(load_audio(audio[i]) if isinstance(audio[i], str) else audio[i],
                                                 wfh, wfw, ignore_shift=True)
                else:
                    if isinstance(audio, str):
                        wf = load_audio_waveform_img(load_audio(audio[i]), wfh, wfw, ignore_shift=True)
                    else:
                        all_silent = True

                if not all_silent:
                    all_silent = not wf.any()
                if all_silent:
                    warnings.warn(f'Audio {i} appears to be entirely silent. suppress_silence will be set to False',
                                  stacklevel=2)
                    batch_suppress_silence[i] = False

    upper_quantile = decode_options.pop('upper_quantile', 0.85)
    lower_quantile = decode_options.pop('lower_quantile', 0.15)
    lower_threshold = decode_options.pop('lower_threshold', 0.15)

    num_frames = [mel.shape[-1] for mel in mels]

    def check_cursors(seekers: List[int], num_frames: List[int]) -> bool:
        """Return False when all seekers have exhausted the length of their audio clips."""
        return any([seeker < nf for seeker, nf in list(zip(seekers, num_frames))])

    with tqdm(total=num_frames, unit='frames', disable=(print_unstab or not pbar)) as tqdm_pbar:
        def update_pbar():
            if not tqdm_pbar.disable:
                midx = num_frames.index(max(num_frames))
                tqdm_pbar.update(min(num_frames[midx], seekers[midx]) - tqdm_pbar.n)

        while check_cursors(seekers, num_frames):
            continue_processing = [seeker < nf for seeker, nf in list(zip(seekers, num_frames))]
            # Only those segments for clips that are not done being processed
            imap = [i for i, v in enumerate(continue_processing) if v]
            batch_segments = []
            batch_segment_durations = []
            batch_timestamp_offsets = []
            batch_suppress_ts_mask = []
            batch_segment_max_ts = []

            for i, mel in enumerate(mels):
                if continue_processing[i]:

                    timestamp_offset = float(seekers[i] * HOP_LENGTH / SAMPLE_RATE)
                    batch_timestamp_offsets.append(timestamp_offset)
                    remaining_duration = float((mel.shape[-1] -seekers[i]) * HOP_LENGTH / SAMPLE_RATE)
                    segment = pad_or_trim(mel[:, :, seekers[i]:], N_FRAMES).to(model.device).to(dtype)
                    batch_segments.append(segment)
                    segment_duration = min(float(segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE), remaining_duration)
                    batch_segment_durations.append(segment_duration)
                    segment_max_ts = segment_duration / time_precision
                    batch_segment_max_ts.append(segment_max_ts)

                    if batch_suppress_silence[i]:
                        wf_seek = int(seekers[i] * ts_scale)
                        segment_wf = wf[..., wf_seek:wf_seek + 1501]
                        if remove_background and \
                                (1 - segment_wf.sum(0).clip(max=1).mean()) < silence_threshold:
                            segment_wf = remove_lower_quantile(segment_wf.astype(np.float32),
                                                               upper_quantile=upper_quantile,
                                                               lower_quantile=lower_quantile,
                                                               lower_threshold=lower_threshold)
                        segment_wf = pad_or_trim(segment_wf, 1501)
                        suppress_ts_mask = torch.from_numpy(wave_to_ts_filter(segment_wf,
                                                                              suppress_middle=suppress_middle,
                                                                              max_index=int(segment_max_ts)))
                        if suppress_ts_mask.all():  # segment is silent
                            seekers[i] += segment.shape[-1]  # fast-forward to the next segment boundary
                            batch_suppress_ts_mask.append(suppress_ts_mask)
                            update_pbar()
                            continue
                    else:
                        suppress_ts_mask = None
                    batch_suppress_ts_mask.append(suppress_ts_mask)
                else:
                    continue

            decode_options["prompt"] = [all_tokens[imap[i]][prompt_reset_since[imap[i]]:] for i in range(len(batch_segments))]
            decode_options["language"] = [l for i, l in enumerate(languages) if continue_processing[i]]

            results, finalized_ts_tokens, ts_logits, tc = decode_with_fallback(torch.stack(batch_segments),
                                                                          suppress_ts_mask=torch.stack(batch_suppress_ts_mask))

            batch_tokens = [torch.tensor(result.tokens) for result in results]
            batch_finalized_ts_tokens = [torch.tensor(finalized_ts_token) for finalized_ts_token in finalized_ts_tokens]
            batch_ts_logits = [torch.tensor(ts_logit) for ts_logit in ts_logits]
            batch_tc = [tc_slice for tc_slice in tc]

            for i, result in enumerate(results):
                if no_speech_threshold is not None:
                    # no voice activity check
                    # print('no_caption_prob', _get_new_attrs(result, 'no_caption_prob'), "no_speech_threshold",
                    #       no_speech_threshold)
                    # Todo adapt to batch inference
                    should_skip = _get_new_attrs(result, 'no_caption_prob')[0] > no_speech_threshold
                    if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                        # don't skip if the logprob is high enough, despite the no_speech_prob
                        should_skip = False

                    if should_skip:
                        seekers[imap[i]] += segment.shape[-1]  # fast-forward to the next segment boundary
                        continue

            batch_timestamp_tokens: List[torch.Tensor] = [tokens.ge(tokenizers[languages[imap[i]]].timestamp_begin)
                                                          for i, tokens in enumerate(batch_tokens)]
            batch_consecutive = [torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1) for
                                 timestamp_tokens in batch_timestamp_tokens]

            for i, consecutive in enumerate(batch_consecutive):
                if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                    last_slice = 0
                    for current_slice in consecutive:
                        sliced_tokens = batch_tokens[i][last_slice:current_slice]
                        sliced_ts_tokens = batch_finalized_ts_tokens[i][last_slice:current_slice]
                        sliced_ts_logits = batch_ts_logits[i][last_slice:current_slice]
                        sliced_tc = batch_tc[i][last_slice:current_slice]
                        start_timestamp_position = (
                                sliced_tokens[0].item() - tokenizers[languages[imap[i]]].timestamp_begin
                        )
                        end_timestamp_position = (
                                sliced_tokens[-1].item() - tokenizers[languages[imap[i]]].timestamp_begin
                        )

                        word_ts = batch_timestamp_offsets[i] + sliced_ts_tokens * time_precision

                        add_segment(
                            seeker=seekers[imap[i]],
                            offset=batch_timestamp_offsets[i],
                            start=batch_timestamp_offsets[i] + start_timestamp_position * time_precision,
                            end=min(batch_timestamp_offsets[i] + end_timestamp_position * time_precision,
                                    batch_timestamp_offsets[i] + batch_segment_durations[i]),
                            text_tokens=sliced_tokens[1:-1],
                            result=results[i],
                            tokenizer=tokenizers[languages[imap[i]]],
                            segments=all_segments[imap[i]],
                            start_timestamps=word_ts[0].tolist(),
                            end_timestamps=word_ts[-1].tolist(),
                            word_timestamps=word_ts[1:-1],
                            start_ts_logits=sliced_ts_logits[0].tolist(),
                            end_ts_logits=sliced_ts_logits[-1].tolist(),
                            word_ts_logits=sliced_ts_logits[1:-1],
                            tc_logits=sliced_tc[1:-1],
                        )
                        last_slice = current_slice
                    last_timestamp_position = (
                        min(batch_tokens[i][last_slice - 1].item() - tokenizers[languages[imap[i]]].timestamp_begin, batch_segment_max_ts[i])
                    )
                    seekers[imap[i]] += last_timestamp_position * input_stride
                    all_tokens[imap[i]].extend(batch_tokens[i][: last_slice + 1].tolist())
                else:
                    duration = batch_segment_durations[i]
                    timestamps = batch_tokens[i][batch_timestamp_tokens[i].nonzero().flatten()]
                    if len(timestamps) > 0:
                        # no consecutive timestamps but it has a timestamp; use the last one.
                        # single timestamp at the end means no speech after the last timestamp.
                        last_timestamp_position = min(timestamps[-1].item() - tokenizers[languages[imap[i]]].timestamp_begin, batch_segment_max_ts[i])
                        duration = last_timestamp_position * time_precision

                    word_ts = batch_timestamp_offsets[i] + batch_finalized_ts_tokens[i] * time_precision

                    add_segment(
                        seeker=seekers[imap[i]],
                        offset=batch_timestamp_offsets[i],
                        start=batch_timestamp_offsets[i],
                        end=batch_timestamp_offsets[i] + duration,
                        text_tokens=batch_tokens[i],
                        result=results[i],
                        tokenizer=tokenizers[languages[imap[i]]],
                        segments=all_segments[imap[i]],
                        word_timestamps=word_ts,
                        word_ts_logits=batch_ts_logits[i],
                        tc_logits=batch_tc[i],
                    )

                    seekers[imap[i]] += segments[imap[i]].shape[-1]
                    all_tokens[imap[i]].extend(batch_tokens[i].tolist())

                if all_segments[imap[i]]:
                    all_segments[imap[i]][-1]['anchor_point'] = True
                    all_segments[imap[i]][-1]['next_offset'] = float(seekers[imap[i]] * HOP_LENGTH / SAMPLE_RATE)
                if not condition_on_previous_text or result.temperature > 0.5:
                    # do not feed the prompt tokens if a high temperature was used
                    prompt_reset_since[imap[i]] = len(all_tokens[imap[i]])

                update_pbar()

    if len(all_segments[imap[i]]) > 1 and all_segments[imap[i]][-1]['alt_start_timestamps'] is None:
        all_segments[imap[i]][-1]['alt_start_timestamps'] = all_segments[imap[i]][-2]['alt_end_timestamps']

    # # my prepare confidence
    # all_segments = my_prepare_confidence_and_words(tokenizer, all_segments)  # side effect

    if stab:
        all_segments[imap[i]] = stabilize_timestamps(all_segments[imap[i]], top_focus=top_focus)
        add_whole_word_ts(tokenizers[languages[imap[i]]], all_segments[imap[i]],
                          merge_non_space=True,
                          prepend_punctuations=prepend_punctuations,
                          append_punctuations=append_punctuations)
        # if verbose:
        #     print('\nSTABILIZED:')
        #     for seg_ in all_segments:
        #         print(f'[{format_timestamp(seg_["start"])} --> {format_timestamp(seg_["end"])}] "{seg_["text"]}"')
        #         if seg_['word_timestamps']:
        #             ts_str = (f' ->[{format_timestamp(ts_["timestamp"])}] "{ts_["word"].strip()}"' for ts_ in
        #                       seg_['word_timestamps'])
        #             print('\n'.join(ts_str), end='\n\n')
        # import pickle
        # with open('filename.pickle', 'wb') as fh:
        #     pickle.dump(all_segments, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return [dict(text=tokenizers[languages[i]].decode(
        [token for token in all_tokens[i][len(initial_prompt):] if token < tokenizers[languages[i]].eot]),
                 segments=all_segments[i], language=languages[i]) for i in range(len(all_segments))]

def _suppress_ts(ts_logits: Tensor, suppress_ts_mask: Tensor = None):
    if suppress_ts_mask is not None:
        ts_logits[:, suppress_ts_mask] = -np.inf


def _ts_topk(ts_logits: Tensor, k: int, prev_ts: Tensor = None) -> Tensor:
    temp_ts = torch.stack(torch.topk(ts_logits, k, dim=-1), 0).unsqueeze(-2)
    return temp_ts if prev_ts is None else torch.cat([prev_ts, temp_ts], dim=-2)


def cli():
    from . import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    from . import load_model
    model = load_model(model_name, device=device, download_root=model_dir)

    for audio_path in args.pop("audio"):
        result = transcribe(model, audio_path, temperature=temperature, **args)

        audio_basename = os.path.basename(audio_path)

        # save TXT
        with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
            write_txt(result["segments"], file=txt)

        # save VTT
        with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
            write_vtt(result["segments"], file=vtt)

        # save SRT
        with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)


if __name__ == '__main__':
    cli()
