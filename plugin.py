import gradio as gr
import re
import os
import json
from typing import Dict, Any, List
from shared.utils.plugins import WAN2GPPlugin

TARGET_MODEL_ARCHITECTURE = "minimax_h3_ref2va"
SETTINGS_FILE_NAME = "settings.json"

SECTION_KEYS = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music"
]

SECTION_DISPLAY_NAMES = {
    "subject_definitions": "1. Subject Definitions (subject_definitions)",
    "summary": "2. Task Summary & Overview (summary)",
    "retention_analysis": "3. Retention Analysis (retention_analysis)",
    "detailed_description": "4. Detailed Description (detailed_description)",
    "overall_soundscape": "5. Overall Soundscape (overall_soundscape)",
    "non_diegetic_music": "6. Non-Diegetic Music (non_diegetic_music)",
}

SECTION_PLACEHOLDERS = {
    "subject_definitions": "<Subject 1> is the young woman in <Picture 1>, preserving her identity, face, hairstyle, and blue cardigan.\n<Audio 1> is the voice-timbre reference for <Subject 1> (S1).",
    "summary": "[reference generation + audio reference] The target video shows <Subject 1> entering a cafe and speaking in one single take.",
    "retention_analysis": "<Subject 1> (appears in [Shot 1]): fully_preserved - identity, hairstyle, clothing, and accessories remain unchanged.\n<Audio 1>: reference - timbre guides dialogue delivery without copying the raw signal.",
    "detailed_description": "The target video uses a cinematic, warm indoor aesthetic.\n[Shot 1] A steady medium shot frames <Subject 1> (S1) as she turns toward the window, saying clearly, <d>[English] Some journeys begin when the map runs out.</d>",
    "overall_soundscape": "Soft indoor cafe ambience, faint rain on the glass, and clear room tone.",
    "non_diegetic_music": "A restrained acoustic guitar motif resolving softly as the shot ends."
}

EXAMPLE_REWRITE = {
    "subject_definitions": "<Subject 1> is the person in <Picture 1>, preserving their exact identity, facial features, skin tone, hairstyle, body proportions, clothing, footwear, and distinctive accessories.",
    "summary": "[reference generation] Place <Subject 1> on a deserted midnight railway platform where a paper bird awakens the station clock during one five-second shot.",
    "retention_analysis": "<Subject 1> (appears in [Shot 1]): fully_preserved - their identity, face, hair, proportions, clothing, footwear, and visible accessories remain recognizable and unchanged; only the location, lighting, action, and paper-bird prop are new.",
    "detailed_description": "The target video is a five-second cinematic magical-realist single take on an abandoned railway platform before dawn. [Shot 1] <Subject 1> stands beneath a monumental stopped clock, framed in a steady medium close-up that keeps their recognizable face and exact appearance from <Picture 1> clearly visible. Cold blue light outlines the iron platform while a warm lamp illuminates their natural skin texture, hair, clothing, and accessories without changing them. A tiny folded paper bird rests on their raised palm. As its paper wings open, the camera gently pushes closer and <Subject 1> watches it with quiet resolve, saying clearly (S1) <d>[English] Some journeys begin when the map runs out.</d> During the final words, the bird lifts a few inches, lands on the clock's minute hand, and the mechanism advances with one solid click. <Subject 1> follows it with their eyes and gives a small hopeful smile. The spoken line and simple transformation complete naturally within the five-second shot; identity, lip motion, gaze, hand position, posture, and paper movement remain stable and physically coherent.",
    "overall_soundscape": "Spacious nighttime station ambience, faint wind through ironwork, a soft paper-wing flutter crossing the stereo field, the synchronized spoken line, one heavy clockwork click, and a distant rail hum.",
    "non_diegetic_music": "A minimal celesta phrase of three notes, resolving softly as the clock moves."
}

DEFAULT_SETTINGS = {
    "active_refs_display_mode": "text",  # "text" or "graphic"
    "graphic_card_size": "medium",      # "small", "medium", "large"
    "sync_from_combined_prompt": False,
}


def get_settings_path() -> str:
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(plugin_dir, SETTINGS_FILE_NAME)


def load_settings() -> Dict[str, Any]:
    path = get_settings_path()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    settings = DEFAULT_SETTINGS.copy()
                    settings.update(data)
                    return settings
        except Exception as e:
            print(f"[MiniMax Ref2VA Studio] Error reading settings: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> None:
    path = get_settings_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"[MiniMax Ref2VA Studio] Error saving settings: {e}")


def is_minimax_model(model_name: str, base_type: str = "") -> bool:
    name_check = TARGET_MODEL_ARCHITECTURE in str(model_name or "").lower()
    base_check = TARGET_MODEL_ARCHITECTURE in str(base_type or "").lower()
    return name_check or base_check


def parse_multisection_prompt(raw_text: str) -> Dict[str, str]:
    result = {k: "" for k in SECTION_KEYS}
    if not raw_text or not raw_text.strip():
        return result

    text = raw_text.strip().replace("\r\n", "\n").replace("\r", "\n")
    pattern = r"(?im)^(subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music)\s*:\s*"
    
    matches = list(re.finditer(pattern, text))
    if not matches:
        result["detailed_description"] = text
        return result

    for i, match in enumerate(matches):
        sec_name = match.group(1).lower()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start_idx:end_idx].strip()
        if sec_name in result:
            result[sec_name] = content

    return result


def assemble_multisection_prompt(
    subject_defs: str,
    summary: str,
    retention: str,
    detailed: str,
    soundscape: str,
    music: str
) -> str:
    sections = [
        ("subject_definitions", subject_defs),
        ("summary", summary),
        ("retention_analysis", retention),
        ("detailed_description", detailed),
        ("overall_soundscape", soundscape),
        ("non_diegetic_music", music),
    ]

    blocks = []
    for key, val in sections:
        val_clean = (val or "").strip()
        if val_clean:
            blocks.append(f"{key}:\n{val_clean}")

    return "\n\n".join(blocks) if blocks else ""


class MiniMaxRef2VAHelperPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "MiniMax Ref2VA Prompt Studio"
        self.version = "1.0.4"
        self.description = "Splits prompts into symmetrical section editors, provides live active reference previews (text & interactive media cards), hover popups, tag insertion palette, and persistent settings."
        self.type = ["extension"]

    def setup_ui(self):
        self.request_component("prompt")
        self.request_component("prompt_column_advanced")
        self.request_component("image_refs")
        self.request_component("image_start")
        self.request_component("image_end")
        self.request_component("video_guide")
        self.request_component("video_source")
        self.request_component("audio_guide")
        self.request_component("audio_guide2")
        self.request_component("settings_file")
        self.request_component("model_choice_target")
        self.request_component("refresh_form_trigger")
        self.request_component("state")
        self.request_global("get_base_model_type")
        self.request_global("get_model_def")
        self.request_global("server_config")

        self.add_custom_js("""
        (function() {
            window._activeRef2VARichEditor = null;
            window._ref2vaSyncingFromGradio = false;
            window._ref2vaHidePreviewTimer = null;
            window._ref2vaCurrentPreviewTag = null;
            window._ref2vaSettings = {
                active_refs_display_mode: 'text',
                graphic_card_size: 'medium',
                sync_from_combined_prompt: false
            };

            function escapeHtml(str) {
                return (str || '')
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            }

            function cancelHidePreview() {
                if (window._ref2vaHidePreviewTimer) {
                    clearTimeout(window._ref2vaHidePreviewTimer);
                    window._ref2vaHidePreviewTimer = null;
                }
            }

            function scheduleHidePreview() {
                cancelHidePreview();
                window._ref2vaHidePreviewTimer = setTimeout(() => {
                    const popup = document.getElementById('ref2va-floating-preview-popup');
                    if (popup) {
                        popup.remove();
                        window._ref2vaCurrentPreviewTag = null;
                    }
                }, 320);
            }

            window.tagToInlineBadgeHtml = function(tagStr) {
                const clean = tagStr.trim();
                let cls = 'ref2va-badge-generic';

                if (/^<Subject/i.test(clean)) { cls = 'ref2va-badge-subject'; }
                else if (/^<Picture/i.test(clean)) { cls = 'ref2va-badge-picture'; }
                else if (/^<Video/i.test(clean)) { cls = 'ref2va-badge-video'; }
                else if (/^<Audio/i.test(clean)) { cls = 'ref2va-badge-audio'; }
                else if (/^\\(S/i.test(clean)) { cls = 'ref2va-badge-speaker'; }
                else if (/^<d>/i.test(clean)) { cls = 'ref2va-badge-dialogue'; }
                else if (/^\\[Shot/i.test(clean)) { cls = 'ref2va-badge-shot'; }
                else if (/^(fully_preserved|partially_preserved|attribute_transfer|weak_reference|fully_copy|partially_copy|reference)$/i.test(clean)) { cls = 'ref2va-badge-retention'; }
                else if (/^\\[(reference generation|keyframe completion|video editing|video continuation|audio reuse|audio reference)\\]/i.test(clean)) { cls = 'ref2va-badge-task'; }
                else if (/^<(scenetrans|cutoff)>/i.test(clean)) { cls = 'ref2va-badge-trans'; }

                const esc = escapeHtml(clean);
                return `<span class="ref2va-inline-badge ${cls}" contenteditable="true" data-raw-tag="${esc}" onmouseenter="window.onTagMouseEnter('${esc}', this, event)" onmouseleave="window.onTagMouseLeave(event)">${esc}</span>`;
            };

            window.onTagMouseEnter = function(tagStr, el, e) {
                cancelHidePreview();
                window.showRef2VAPreview(tagStr, el, e);
            };

            window.onTagMouseLeave = function(e) {
                scheduleHidePreview();
            };

            window.plainTextToRichHtml = function(text) {
                if (!text) return '<div><br></div>';
                const lines = text.split('\\n');
                return lines.map(line => {
                    let esc = escapeHtml(line);
                    esc = esc.replace(/&lt;(Subject|Picture|Video|Audio)\s+(\d+)&gt;/gi, (m, k, n) => window.tagToInlineBadgeHtml(`<${k} ${n}>`));
                    esc = esc.replace(/\((S\d+)\)/gi, (m, s) => window.tagToInlineBadgeHtml(`(${s})`));
                    esc = esc.replace(/\[Shot\s+(\d+)([^\]]*)\]/gi, (m, n, rest) => window.tagToInlineBadgeHtml(`[Shot ${n}${rest}]`));
                    esc = esc.replace(/&lt;d&gt;([\s\S]*?)&lt;\/d&gt;/gi, (m, diag) => window.tagToInlineBadgeHtml(`<d>${diag}</d>`));
                    esc = esc.replace(/&lt;(scenetrans|cutoff)&gt;/gi, (m, t) => window.tagToInlineBadgeHtml(`<${t}>`));
                    esc = esc.replace(/\\b(fully_preserved|partially_preserved|attribute_transfer|weak_reference|fully_copy|partially_copy|reference)\\b/gi, (m) => window.tagToInlineBadgeHtml(m));
                    esc = esc.replace(/\[(reference generation|keyframe completion|video editing|video continuation|audio reuse|audio reference)\]/gi, (m, t) => window.tagToInlineBadgeHtml(`[${t}]`));
                    return `<div>${esc || '<br>'}</div>`;
                }).join('');
            };

            window.extractPlainTextFromRichEditor = function(node) {
                let text = '';
                for (let child of node.childNodes) {
                    if (child.nodeType === Node.TEXT_NODE) {
                        text += child.textContent;
                    } else if (child.nodeType === Node.ELEMENT_NODE) {
                        if (child.classList && child.classList.contains('ref2va-inline-badge')) {
                            text += child.textContent.trim();
                        } else if (child.tagName === 'BR') {
                            text += '\\n';
                        } else if (child.tagName === 'DIV' || child.tagName === 'P') {
                            const inner = window.extractPlainTextFromRichEditor(child);
                            text += (text.length > 0 && !text.endsWith('\\n') ? '\\n' : '') + inner;
                        } else {
                            text += window.extractPlainTextFromRichEditor(child);
                        }
                    }
                }
                return text;
            };

            window.syncRichEditorToGradio = function(editor) {
                if (window._ref2vaSyncingFromGradio) return;
                const container = editor.closest('.ref2va-rich-field-wrapper');
                if (!container) return;

                const textarea = container.querySelector('.ref2va-hidden-gradio-input textarea');
                const plainText = window.extractPlainTextFromRichEditor(editor).trimEnd();

                if (textarea && textarea.value !== plainText) {
                    const proto = window.HTMLTextAreaElement.prototype;
                    const nativeValueSetter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                    nativeValueSetter.call(textarea, plainText);
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                }

                window.assembleAndSyncAllSections();
            };

            window.assembleAndSyncAllSections = function() {
                const secNames = ['subject_definitions', 'summary', 'retention_analysis', 'detailed_description', 'overall_soundscape', 'non_diegetic_music'];
                const blocks = [];

                secNames.forEach(secName => {
                    const editor = document.querySelector(`.ref2va-editor-${secName}`);
                    if (editor) {
                        const val = window.extractPlainTextFromRichEditor(editor).trim();
                        if (val) {
                            blocks.push(`${secName}:\\n${val}`);
                        }
                    }
                });

                const assembled = blocks.join('\\n\\n');
                const mainPromptTextarea = document.querySelector('#wangp-prompt-advanced textarea') || document.querySelector('#wangp-prompt-advanced');
                if (mainPromptTextarea && mainPromptTextarea.value !== assembled) {
                    const proto = window.HTMLTextAreaElement.prototype;
                    const nativeValueSetter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                    nativeValueSetter.call(mainPromptTextarea, assembled);
                    mainPromptTextarea.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };

            window.updateRichEditorsFromTextareas = function(force = false) {
                const secNames = ['subject_definitions', 'summary', 'retention_analysis', 'detailed_description', 'overall_soundscape', 'non_diegetic_music'];
                secNames.forEach(secName => {
                    const wrapper = document.querySelector(`.ref2va-wrapper-${secName}`);
                    if (!wrapper) return;
                    const textarea = wrapper.querySelector('.ref2va-hidden-gradio-input textarea');
                    const richEditor = wrapper.querySelector(`.ref2va-editor-${secName}`);
                    if (!textarea || !richEditor) return;

                    if (!force && document.activeElement === richEditor) {
                        return;
                    }

                    const currentPlain = window.extractPlainTextFromRichEditor(richEditor).trim();
                    const targetPlain = (textarea.value || '').trim();

                    if (!force && currentPlain === targetPlain) {
                        return;
                    }

                    richEditor.innerHTML = window.plainTextToRichHtml(textarea.value);
                });
            };

            window.insertRef2VAText = function(tagStr, isWrap = false, wrapPrefix = '', wrapSuffix = '') {
                let editor = window._activeRef2VARichEditor;
                if (!editor || !document.body.contains(editor)) {
                    editor = document.querySelector('.ref2va-editor-detailed_description') || document.querySelector('.ref2va-rich-editor');
                }

                if (!editor) return;
                editor.focus();

                const sel = window.getSelection();
                if (!sel || sel.rangeCount === 0) return;

                const range = sel.getRangeAt(0);
                range.deleteContents();

                let insertedContent = tagStr;
                if (isWrap) {
                    insertedContent = wrapPrefix + (sel.toString() || '...') + wrapSuffix;
                }

                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = window.tagToInlineBadgeHtml(insertedContent);
                const badgeNode = tempDiv.firstChild;

                if (badgeNode) {
                    range.insertNode(badgeNode);
                    const space = document.createTextNode(' ');
                    badgeNode.parentNode.insertBefore(space, badgeNode.nextSibling);

                    range.setStartAfter(space);
                    range.setEndAfter(space);
                    sel.removeAllRanges();
                    sel.addRange(range);
                } else {
                    const textNode = document.createTextNode(insertedContent + ' ');
                    range.insertNode(textNode);
                    range.setStartAfter(textNode);
                    range.setEndAfter(textNode);
                    sel.removeAllRanges();
                    sel.addRange(range);
                }

                window.syncRichEditorToGradio(editor);
            };

            function getOrderedThumbnailsFromContainer(container) {
                if (!container) return [];
                let imgs = Array.from(container.querySelectorAll('.thumbnails img, button.thumbnail-item img, .grid-wrap img, .thumbnail-item img'));
                if (imgs.length === 0) {
                    imgs = Array.from(container.querySelectorAll('img')).filter(i => {
                        return !i.closest('.preview') && !i.closest('#gallery') && !i.closest('#plugin_guides') && !i.closest('.tutorial');
                    });
                }
                const uniqueSrcs = [];
                const seen = new Set();
                for (const img of imgs) {
                    const src = img.currentSrc || img.src || img.getAttribute('src');
                    if (!src) continue;
                    if (src.includes('data:image/svg') || src.includes('/icons/') || src.includes('favicon')) continue;
                    if (!seen.has(src)) {
                        seen.add(src);
                        uniqueSrcs.push(src);
                    }
                }
                return uniqueSrcs;
            }

            function findContainerByKeywords(keywords) {
                const labels = Array.from(document.querySelectorAll('.label, label, span[data-testid="block-label"], .gr-form-label, span.block, .form-label'));
                for (const lbl of labels) {
                    if (lbl.closest('#plugin_guides') || lbl.closest('.tutorial') || lbl.closest('#gallery')) continue;
                    const text = (lbl.textContent || '').toLowerCase();
                    const matched = keywords.some(kw => text.includes(kw.toLowerCase()));
                    if (matched) {
                        return lbl.closest('.gr-block') || lbl.closest('.gr-box') || lbl.closest('.block') || lbl.closest('.gr-form') || lbl.parentElement.parentElement;
                    }
                }
                return null;
            }

            window.findWanGPMediaElement = function(mediaType, slotNum) {
                if (mediaType === 'picture') {
                    const startContainer = findContainerByKeywords(['starting points', 'start image', 'start with image']);
                    const refsContainer = findContainerByKeywords(['reference image', 'reference images', 'inject reference']);
                    const endContainer = findContainerByKeywords(['end image', 'ending points']);

                    const startImgs = getOrderedThumbnailsFromContainer(startContainer);
                    const refImgs = getOrderedThumbnailsFromContainer(refsContainer);
                    const endImgs = getOrderedThumbnailsFromContainer(endContainer);

                    const hasStart = startImgs.length > 0;

                    if (slotNum === 1 && hasStart) {
                        return { src: startImgs[0], type: 'image', desc: 'Start Image Anchor' };
                    }

                    const targetRefIdx = hasStart ? (slotNum - 2) : (slotNum - 1);
                    if (targetRefIdx >= 0 && targetRefIdx < refImgs.length) {
                        return { src: refImgs[targetRefIdx], type: 'image', desc: `Reference Image #${targetRefIdx + 1}` };
                    }

                    if (endImgs.length > 0 && slotNum === (hasStart ? 2 + refImgs.length : 1 + refImgs.length)) {
                        return { src: endImgs[0], type: 'image', desc: 'End Image Anchor' };
                    }
                } else if (mediaType === 'video') {
                    const srcContainer = findContainerByKeywords(['video to continue', 'video source', 'source video']);
                    const guideContainer = findContainerByKeywords(['control video', 'video guide']);
                    const guide2Container = findContainerByKeywords(['control video 2', 'video guide 2']);

                    const validContainers = [srcContainer, guideContainer, guide2Container].filter(Boolean);
                    const discoveredVideos = [];

                    for (const cont of validContainers) {
                        if (cont.closest('#plugin_guides') || cont.closest('.tutorial-video')) continue;
                        const vids = Array.from(cont.querySelectorAll('video')).filter(v => {
                            return !v.closest('#gallery') && !v.closest('.ref2va-preview-card') && !v.closest('.ref2va-active-refs-mount');
                        });
                        for (const v of vids) {
                            const src = v.currentSrc || v.src || (v.querySelector('source') ? v.querySelector('source').src : null);
                            if (src && !src.includes('tutorial') && !src.includes('guide')) {
                                discoveredVideos.push({ src: src, type: 'video' });
                            }
                        }
                    }

                    if (discoveredVideos.length === 0) {
                        const genTab = document.querySelector('#media_gen') || document.querySelector('#tab_media_gen') || document.body;
                        const vids = Array.from(genTab.querySelectorAll('#video_input video, #video_input2 video')).filter(v => {
                            return !v.closest('#plugin_guides') && !v.closest('#gallery');
                        });
                        for (const v of vids) {
                            const src = v.currentSrc || v.src || (v.querySelector('source') ? v.querySelector('source').src : null);
                            if (src && !src.includes('tutorial')) {
                                discoveredVideos.push({ src: src, type: 'video' });
                            }
                        }
                    }

                    if (slotNum >= 1 && slotNum <= discoveredVideos.length) {
                        return discoveredVideos[slotNum - 1];
                    }
                } else if (mediaType === 'audio') {
                    const allAudioBlocks = Array.from(document.querySelectorAll('.audio-container, [data-testid="audio"], .gr-audio')).filter(el => {
                        return !el.closest('#gallery') && !el.closest('#audio') && !el.closest('.ref2va-preview-card') && !el.closest('.ref2va-active-refs-mount') && !el.closest('#plugin_guides');
                    });

                    const targetBlock = allAudioBlocks[slotNum - 1];
                    if (targetBlock) {
                        const a = targetBlock.querySelector('audio');
                        if (a) {
                            const src = a.currentSrc || a.src || (a.querySelector('source') ? a.querySelector('source').src : null);
                            if (src) return { src: src, type: 'audio', desc: `Audio Track ${slotNum}` };
                        }
                        const link = targetBlock.querySelector('a[download], a[href*="/file="], a[href*="/gradio_api/file="], a[href*="blob:"]');
                        if (link && link.href) return { src: link.href, type: 'audio', desc: `Audio Track ${slotNum}` };
                    }

                    const allAudios = Array.from(document.querySelectorAll('audio')).filter(a => {
                        return !a.closest('#gallery') && !a.closest('#audio') && !a.closest('.ref2va-preview-card') && !a.closest('.ref2va-active-refs-mount') && !a.closest('#plugin_guides');
                    });
                    const targetAudio = allAudios[slotNum - 1];
                    if (targetAudio) {
                        const src = targetAudio.currentSrc || targetAudio.src || (targetAudio.querySelector('source') ? targetAudio.querySelector('source').src : null);
                        if (src) return { src: src, type: 'audio', desc: `Audio Track ${slotNum}` };
                    }
                }
                return null;
            };

            window.showRef2VAPreview = function(tagStr, triggerEl, event) {
                cancelHidePreview();
                const rawTag = tagStr.trim();

                const existing = document.getElementById('ref2va-floating-preview-popup');
                if (existing && window._ref2vaCurrentPreviewTag === rawTag) {
                    return;
                }
                if (existing) existing.remove();

                window._ref2vaCurrentPreviewTag = rawTag;

                const popup = document.createElement('div');
                popup.id = 'ref2va-floating-preview-popup';
                popup.className = 'ref2va-preview-card';
                popup.addEventListener('mouseenter', cancelHidePreview);
                popup.addEventListener('mouseleave', scheduleHidePreview);
                popup.addEventListener('click', (e) => e.stopPropagation());

                const closeBtn = document.createElement('button');
                closeBtn.className = 'ref2va-popup-close-btn';
                closeBtn.innerHTML = '×';
                closeBtn.onclick = () => {
                    popup.remove();
                    window._ref2vaCurrentPreviewTag = null;
                };
                popup.appendChild(closeBtn);

                const header = document.createElement('div');
                header.className = 'ref2va-popup-header';
                popup.appendChild(header);

                const body = document.createElement('div');
                body.className = 'ref2va-popup-body';

                if (/^<Picture\s+\d+>/i.test(rawTag)) {
                    const num = parseInt(rawTag.match(/\d+/)[0], 10);
                    header.innerHTML = `🖼️ <strong>&lt;Picture ${num}&gt;</strong> Reference Image`;
                    const media = window.findWanGPMediaElement('picture', num);

                    if (media && media.src) {
                        const img = document.createElement('img');
                        img.src = media.src;
                        img.className = 'ref2va-preview-media';
                        body.appendChild(img);
                    } else {
                        body.innerHTML = `<span class="ref2va-popup-muted">No image uploaded in slot &lt;Picture ${num}&gt;.</span>`;
                    }
                } else if (/^<Video\s+\d+>/i.test(rawTag)) {
                    const num = parseInt(rawTag.match(/\d+/)[0], 10);
                    header.innerHTML = `🎥 <strong>&lt;Video ${num}&gt;</strong> Reference Video`;
                    const media = window.findWanGPMediaElement('video', num);

                    if (media && media.src) {
                        const v = document.createElement('video');
                        v.src = media.src;
                        v.controls = true;
                        v.autoplay = true;
                        v.preload = "auto";
                        v.loop = true;
                        v.className = 'ref2va-preview-media';
                        body.appendChild(v);
                        v.play().catch(() => {});
                    } else {
                        body.innerHTML = `<span class="ref2va-popup-muted">No video loaded in slot &lt;Video ${num}&gt;.</span>`;
                    }
                } else if (/^<Audio\s+\d+>/i.test(rawTag)) {
                    const num = parseInt(rawTag.match(/\d+/)[0], 10);
                    header.innerHTML = `🔊 <strong>&lt;Audio ${num}&gt;</strong> Reference Track`;
                    const media = window.findWanGPMediaElement('audio', num);

                    if (media && media.src) {
                        const a = document.createElement('audio');
                        a.src = media.src;
                        a.controls = true;
                        a.autoplay = true;
                        a.preload = "auto";
                        a.style.width = '100%';
                        body.appendChild(a);
                        a.play().catch(() => {});
                    } else {
                        body.innerHTML = `<span class="ref2va-popup-muted">No audio track loaded for &lt;Audio ${num}&gt;. Upload an audio file above.</span>`;
                    }
                } else if (/^<Subject\s+\d+>/i.test(rawTag)) {
                    const num = parseInt(rawTag.match(/\d+/)[0], 10);
                    header.innerHTML = `👤 <strong>&lt;Subject ${num}&gt;</strong> Definition`;

                    const subjEditor = document.querySelector('.ref2va-editor-subject_definitions');
                    let defText = "";
                    if (subjEditor) {
                        const lines = window.extractPlainTextFromRichEditor(subjEditor).split('\\n');
                        for (let line of lines) {
                            if (line.toLowerCase().includes(`<subject ${num}>`)) {
                                defText = line.trim();
                                break;
                            }
                        }
                    }

                    if (defText) {
                        const desc = document.createElement('div');
                        desc.className = 'ref2va-popup-desc';
                        desc.textContent = defText;
                        body.appendChild(desc);

                        const picRefMatch = defText.match(/<Picture\\s+(\\d+)>/i);
                        if (picRefMatch) {
                            const pNum = parseInt(picRefMatch[1], 10);
                            const pMedia = window.findWanGPMediaElement('picture', pNum);
                            if (pMedia && pMedia.src) {
                                const miniImg = document.createElement('img');
                                miniImg.src = pMedia.src;
                                miniImg.className = 'ref2va-preview-media';
                                miniImg.style.marginTop = '6px';
                                body.appendChild(miniImg);
                            }
                        }
                    } else {
                        body.innerHTML = `<span class="ref2va-popup-muted">No definition for &lt;Subject ${num}&gt; found in <b>subject_definitions</b>.</span>`;
                    }
                } else if (/^\[Shot\s+\d+/i.test(rawTag)) {
                    header.innerHTML = `🎬 <strong>${rawTag}</strong> Shot Breakdown`;
                    const detEditor = document.querySelector('.ref2va-editor-detailed_description');
                    let shotText = "";
                    if (detEditor) {
                        const val = window.extractPlainTextFromRichEditor(detEditor);
                        const shotIdx = val.indexOf(rawTag);
                        if (shotIdx !== -1) {
                            const nextShotIdx = val.indexOf('[Shot ', shotIdx + rawTag.length);
                            shotText = (nextShotIdx !== -1) ? val.substring(shotIdx, nextShotIdx) : val.substring(shotIdx);
                        }
                    }

                    if (shotText) {
                        const desc = document.createElement('div');
                        desc.className = 'ref2va-popup-desc';
                        desc.textContent = shotText.trim();
                        body.appendChild(desc);
                    } else {
                        body.innerHTML = `<span class="ref2va-popup-muted">Shot details will appear when written in <b>detailed_description</b>.</span>`;
                    }
                } else if (/^<d>/i.test(rawTag)) {
                    header.innerHTML = `💬 <strong>Dialogue Line</strong>`;
                    const cleanDialogue = rawTag.replace(/<\/?d>/g, '').trim();
                    const words = cleanDialogue.replace(/\[[^\]]+\]/g, '').trim().split(/\\s+/).filter(Boolean);
                    const estimatedSeconds = Math.max(1.5, (words.length / 2.6)).toFixed(1);

                    body.innerHTML = `
                        <div class="ref2va-popup-desc">
                            <div style="font-size:13px; color:#fb7185; font-weight:bold; margin-bottom:4px;">${cleanDialogue}</div>
                            <div style="font-size:11px; color:#94a3b8;">Word count: <b>${words.length}</b> words | Estimated speaking duration: <b>~${estimatedSeconds}s</b></div>
                        </div>
                    `;
                } else if (/^\(S\d+\)/i.test(rawTag)) {
                    header.innerHTML = `🗣️ <strong>${rawTag}</strong> Speaker Profile`;
                    const subjEditor = document.querySelector('.ref2va-editor-subject_definitions');
                    let speakerRole = "";
                    if (subjEditor) {
                        const lines = window.extractPlainTextFromRichEditor(subjEditor).split('\\n');
                        for (let line of lines) {
                            if (line.includes(rawTag)) {
                                speakerRole = line.trim();
                                break;
                            }
                        }
                    }

                    body.innerHTML = `
                        <div class="ref2va-popup-desc">
                            <div><b>Speaker Identifier:</b> ${rawTag}</div>
                            <div style="margin-top:4px; font-size:11px; color:#cbd5e1;">${speakerRole || 'Assigned to vocal timeline in detailed_description.'}</div>
                        </div>
                    `;
                } else if (/^(fully_preserved|partially_preserved|attribute_transfer|weak_reference|fully_copy|partially_copy|reference)$/i.test(rawTag)) {
                    header.innerHTML = `📌 <strong>${rawTag}</strong> Retention Marker`;
                    const descriptions = {
                        'fully_preserved': 'The defined role, identity, facial features, proportions, and clothing of referenced content are completely preserved.',
                        'partially_preserved': 'Content is retained, but certain defined traits (such as costume, pose, or setting) are modified.',
                        'attribute_transfer': 'Characteristics from the reference asset are transferred onto a different target subject.',
                        'weak_reference': 'Retains only broad visual or tonal similarity in composition, category, or atmosphere.',
                        'fully_copy': 'Complete 1:1 reuse of source audio signal as the final target soundtrack.',
                        'reference': 'Audio signal is not copied directly; only voice timbre, delivery, or rhythm is referenced.'
                    };
                    body.innerHTML = `<div class="ref2va-popup-desc">${descriptions[rawTag.toLowerCase()] || 'Retention relationship marker for reference tracking.'}</div>`;
                } else {
                    header.innerHTML = `🎯 <strong>${rawTag}</strong> Task Descriptor`;
                    body.innerHTML = `<div class="ref2va-popup-desc">Formal full-reference task descriptor recognized by MiniMax H3 Ref2VA architecture.</div>`;
                }

                popup.appendChild(body);
                document.body.appendChild(popup);

                const rect = triggerEl.getBoundingClientRect();
                const popupRect = popup.getBoundingClientRect();
                
                let top = rect.top - popupRect.height - 12;
                let left = rect.left + (rect.width / 2) - (popupRect.width / 2);

                if (top < 10) {
                    top = rect.bottom + 12;
                }
                if (left < 10) left = 10;
                if (left + popupRect.width > window.innerWidth - 10) {
                    left = window.innerWidth - popupRect.width - 10;
                }

                popup.style.top = `${top + window.scrollY}px`;
                popup.style.left = `${left + window.scrollX}px`;
            };

            document.addEventListener('click', function(e) {
                const popup = document.getElementById('ref2va-floating-preview-popup');
                if (popup && !popup.contains(e.target) && !e.target.closest('.ref2va-inline-badge') && !e.target.closest('.ref2va-btn-ref') && !e.target.closest('.ref2va-graphic-card')) {
                    popup.remove();
                    window._ref2vaCurrentPreviewTag = null;
                }
            });

            // Live Active Reference Bar Generator (No Hover Popups on Graphic Mode)
            window.refreshActiveReferencesBar = function() {
                const barContainer = document.querySelector('.ref2va-active-refs-mount');
                if (!barContainer) return;

                const settingsInput = document.querySelector('.ref2va-settings-json-input textarea');
                if (settingsInput && settingsInput.value) {
                    try {
                        window._ref2vaSettings = JSON.parse(settingsInput.value);
                    } catch(e) {}
                }

                const displayMode = window._ref2vaSettings.active_refs_display_mode || 'text';
                const cardSize = window._ref2vaSettings.graphic_card_size || 'medium';

                const detected = [];

                for (let i = 1; i <= 6; i++) {
                    const m = window.findWanGPMediaElement('picture', i);
                    if (m && m.src) {
                        detected.push({ tag: `<Picture ${i}>`, label: `🖼️ <Picture ${i}>`, type: 'picture', num: i, media: m });
                    }
                }
                for (let i = 1; i <= 3; i++) {
                    const m = window.findWanGPMediaElement('video', i);
                    if (m && m.src) {
                        detected.push({ tag: `<Video ${i}>`, label: `🎥 <Video ${i}>`, type: 'video', num: i, media: m });
                    }
                }
                for (let i = 1; i <= 2; i++) {
                    const m = window.findWanGPMediaElement('audio', i);
                    if (m && m.src) {
                        detected.push({ tag: `<Audio ${i}>`, label: `🔊 <Audio ${i}>`, type: 'audio', num: i, media: m });
                    }
                }

                // Equality guard: stops infinite flashing cycle
                const cacheKey = detected.map(d => `${d.tag}:${d.media.src}`).join('|') + `|mode:${displayMode}|size:${cardSize}`;
                if (barContainer.dataset.renderedCache === cacheKey) {
                    return;
                }
                barContainer.dataset.renderedCache = cacheKey;

                if (detected.length === 0) {
                    barContainer.innerHTML = '';
                    barContainer.style.display = 'none';
                    return;
                }

                barContainer.style.display = 'flex';
                barContainer.innerHTML = '';

                const labelSpan = document.createElement('span');
                labelSpan.className = 'ref2va-tag-label';
                labelSpan.style.color = '#10b981';
                labelSpan.textContent = '⚡ Active References:';
                barContainer.appendChild(labelSpan);

                const itemsWrapper = document.createElement('div');
                itemsWrapper.className = `ref2va-refs-items-wrapper mode-${displayMode} size-${cardSize}`;
                barContainer.appendChild(itemsWrapper);

                detected.forEach(item => {
                    if (displayMode === 'graphic') {
                        // Graphic Mode: Embedded media with clean Play-only controls; NO HOVER POPUP
                        const card = document.createElement('div');
                        card.className = `ref2va-graphic-card size-${cardSize}`;
                        
                        const mediaWrapper = document.createElement('div');
                        mediaWrapper.className = 'ref2va-graphic-media-wrapper';

                        if (item.type === 'picture') {
                            const img = document.createElement('img');
                            img.src = item.media.src;
                            img.className = 'ref2va-graphic-img';
                            mediaWrapper.appendChild(img);
                            img.onclick = () => window.insertRef2VAText(item.tag);
                        } else if (item.type === 'video') {
                            const vid = document.createElement('video');
                            vid.src = item.media.src;
                            vid.controls = true;
                            vid.autoplay = false;
                            vid.preload = "metadata";
                            vid.className = 'ref2va-graphic-video';
                            mediaWrapper.appendChild(vid);
                        } else if (item.type === 'audio') {
                            const audBox = document.createElement('div');
                            audBox.className = 'ref2va-graphic-audio-box';
                            audBox.innerHTML = '<span style="font-size:16px;">🔊</span>';
                            const aud = document.createElement('audio');
                            aud.src = item.media.src;
                            aud.controls = true;
                            aud.autoplay = false;
                            aud.preload = "metadata";
                            aud.className = 'ref2va-graphic-audio-ctrl';
                            audBox.appendChild(aud);
                            mediaWrapper.appendChild(audBox);
                        }

                        const insertBtn = document.createElement('button');
                        insertBtn.className = 'ref2va-graphic-insert-btn';
                        insertBtn.textContent = item.tag;
                        insertBtn.title = `Insert ${item.tag} into active section`;
                        insertBtn.onclick = () => window.insertRef2VAText(item.tag);

                        card.appendChild(mediaWrapper);
                        card.appendChild(insertBtn);
                        itemsWrapper.appendChild(card);
                    } else {
                        // Text Mode: Text button with live onhover preview popup
                        const btn = document.createElement('button');
                        btn.className = 'ref2va-btn ref2va-btn-ref';
                        btn.textContent = item.label;
                        btn.onclick = () => window.insertRef2VAText(item.tag);
                        btn.onmouseenter = (e) => window.onTagMouseEnter(item.tag, btn, e);
                        btn.onmouseleave = (e) => window.onTagMouseLeave(e);
                        itemsWrapper.appendChild(btn);
                    }
                });
            };

            window.setupAllRichEditors = function() {
                const secNames = ['subject_definitions', 'summary', 'retention_analysis', 'detailed_description', 'overall_soundscape', 'non_diegetic_music'];

                secNames.forEach(secName => {
                    const wrapper = document.querySelector(`.ref2va-wrapper-${secName}`);
                    if (!wrapper) return;

                    const textarea = wrapper.querySelector('.ref2va-hidden-gradio-input textarea');
                    if (!textarea) return;

                    let richEditor = wrapper.querySelector(`.ref2va-editor-${secName}`);
                    if (!richEditor) {
                        richEditor = document.createElement('div');
                        richEditor.className = `ref2va-rich-editor ref2va-editor-${secName}`;
                        richEditor.contentEditable = 'true';
                        richEditor.spellcheck = false;

                        richEditor.innerHTML = window.plainTextToRichHtml(textarea.value);

                        richEditor.addEventListener('focus', () => {
                            window._activeRef2VARichEditor = richEditor;
                        });

                        richEditor.addEventListener('input', () => {
                            window.syncRichEditorToGradio(richEditor);
                        });

                        wrapper.appendChild(richEditor);
                    }
                });
            };

            window.wrapMainPromptInAccordion = function() {
                const mainPromptTextarea = document.querySelector('#wangp-prompt-advanced');
                if (!mainPromptTextarea) return;

                const column = mainPromptTextarea.closest('.wangp-prompt-tools-stack') || mainPromptTextarea.closest('.gr-form') || mainPromptTextarea.parentElement;
                if (!column || column.dataset.ref2vaWrapped === 'true') return;

                const drawer = document.createElement('details');
                drawer.className = 'ref2va-raw-prompt-drawer';
                drawer.open = false;

                const summary = document.createElement('summary');
                summary.className = 'ref2va-raw-prompt-summary';
                summary.innerHTML = '<span>📝 <strong>Combined Main Prompt (Read/Write)</strong></span>';
                drawer.appendChild(summary);

                const content = document.createElement('div');
                content.className = 'ref2va-raw-prompt-content';
                column.parentNode.insertBefore(drawer, column);
                content.appendChild(column);
                drawer.appendChild(content);
                column.dataset.ref2vaWrapped = 'true';
            };

            let _refObserverTimer = null;
            const observer = new MutationObserver((mutations) => {
                const isRelevant = mutations.some(m => {
                    const target = m.target;
                    if (!target || !target.closest) return true;
                    if (target.closest('.ref2va-container') || target.closest('.ref2va-preview-card')) return false;
                    return true;
                });
                if (!isRelevant) return;

                if (_refObserverTimer) clearTimeout(_refObserverTimer);
                _refObserverTimer = setTimeout(() => {
                    window.refreshActiveReferencesBar();
                }, 200);
            });
            observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'value', 'class', 'href'] });

            setTimeout(() => {
                window.wrapMainPromptInAccordion();
                window.setupAllRichEditors();
                window.refreshActiveReferencesBar();
            }, 600);
        })();
        """)

    def post_ui_setup(self, components: Dict[str, gr.components.Component]):
        self.main_prompt = components.get("prompt")
        self.prompt_column_advanced = components.get("prompt_column_advanced")
        self.image_refs = components.get("image_refs")
        self.image_start = components.get("image_start")
        self.image_end = components.get("image_end")
        self.video_guide = components.get("video_guide")
        self.video_source = components.get("video_source")
        self.audio_guide = components.get("audio_guide")
        self.audio_guide2 = components.get("audio_guide2")
        self.settings_file = components.get("settings_file")
        self.model_choice_target = components.get("model_choice_target")
        self.refresh_form_trigger = components.get("refresh_form_trigger")
        self.state_component = components.get("state")

        initial_model = ""
        if hasattr(self, "server_config") and isinstance(self.server_config, dict):
            initial_model = self.server_config.get("last_model_type", "")
        
        initial_visible = is_minimax_model(initial_model)
        loaded_settings = load_settings()

        def create_studio_ui():
            custom_css = """
            <style>
            .ref2va-container {
                border: 1px solid rgba(59, 130, 246, 0.35);
                background: transparent;
                border-radius: 8px;
                padding: 4px 10px 10px 10px !important;
                margin-top: 0px !important;
                margin-bottom: 10px;
            }
            .ref2va-container > div:has(> style),
            .ref2va-container > .gr-html:empty {
                display: none !important;
                margin: 0 !important;
                padding: 0 !important;
                height: 0 !important;
            }
            .ref2va-header-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 6px;
                padding: 0 !important;
                margin: 0 0 4px 0 !important;
                flex-wrap: wrap;
            }
            .ref2va-header-title {
                font-size: 14px;
                font-weight: 700;
                color: #38bdf8;
                margin: 0 !important;
                padding: 0 !important;
                display: flex;
                align-items: center;
                gap: 6px;
                line-height: 1.2;
            }
            .ref2va-settings-drawer {
                border: 1px solid rgba(59, 130, 246, 0.45);
                border-radius: 6px;
                background: var(--input-background-fill, rgba(15, 23, 42, 0.7));
                padding: 8px 10px;
                margin-bottom: 6px;
            }

            .ref2va-active-refs-mount {
                display: flex;
                flex-direction: column;
                gap: 4px;
                margin-top: 0px;
                margin-bottom: 6px;
                padding: 5px 8px;
                border-radius: 6px;
                background: rgba(16, 185, 129, 0.06);
                border: 1px dashed rgba(16, 185, 129, 0.35);
            }
            .ref2va-refs-items-wrapper {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                align-items: center;
            }
            .ref2va-refs-items-wrapper.mode-graphic {
                align-items: flex-start;
            }

            .ref2va-graphic-card {
                display: flex;
                flex-direction: column;
                border: 1px solid rgba(16, 185, 129, 0.5);
                background: rgba(15, 23, 42, 0.85);
                border-radius: 6px;
                padding: 4px;
                position: relative;
                box-sizing: border-box;
                overflow: hidden;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            }
            .ref2va-graphic-card.size-small { width: 95px; height: 95px; }
            .ref2va-graphic-card.size-medium { width: 120px; height: 120px; }
            .ref2va-graphic-card.size-large { width: 150px; height: 150px; }

            .ref2va-graphic-media-wrapper {
                width: 100%;
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                border-radius: 4px;
                background: #020617;
                position: relative;
            }
            .ref2va-graphic-img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                cursor: pointer;
                transition: transform 0.15s ease;
            }
            .ref2va-graphic-img:hover {
                transform: scale(1.05);
            }
            .ref2va-graphic-video {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            .ref2va-graphic-audio-box {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 2px;
                padding: 2px;
            }
            .ref2va-graphic-audio-ctrl {
                width: 96% !important;
                height: 24px !important;
                transform: scale(0.9);
            }
            .ref2va-graphic-insert-btn {
                margin-top: 4px !important;
                width: 100% !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                padding: 3px 6px !important;
                text-align: center !important;
                background: #065f46 !important;
                color: #ffffff !important;
                border: 1px solid #10b981 !important;
                border-radius: 4px !important;
                cursor: pointer !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
                text-shadow: 0 1px 2px rgba(0,0,0,0.6) !important;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                line-height: 1.3;
            }
            .ref2va-graphic-insert-btn:hover {
                background: #059669 !important;
                color: #ffffff !important;
                border-color: #34d399 !important;
            }

            .ref2va-toolbar-group {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-bottom: 4px;
                align-items: center;
            }
            .ref2va-btn {
                font-size: 11px !important;
                padding: 2px 7px !important;
                min-width: unset !important;
                height: 25px !important;
            }
            .ref2va-btn-ref {
                background: rgba(16, 185, 129, 0.15) !important;
                border: 1px solid rgba(16, 185, 129, 0.5) !important;
                color: #10b981 !important;
                font-weight: 600 !important;
            }
            .ref2va-btn-ref:hover {
                background: rgba(16, 185, 129, 0.3) !important;
                color: #ffffff !important;
            }
            .ref2va-tag-label {
                font-size: 11px;
                font-weight: 700;
                color: #3b82f6;
                margin-right: 4px;
            }

            .ref2va-raw-prompt-drawer {
                border: 1px solid var(--border-color-primary, rgba(128, 128, 128, 0.25)) !important;
                border-radius: 6px !important;
                background: transparent !important;
                margin-top: 0px !important;
                margin-bottom: 3px !important;
                overflow: hidden !important;
            }
            .ref2va-raw-prompt-summary {
                padding: 5px 8px !important;
                cursor: pointer !important;
                font-size: 11.5px !important;
                font-weight: 600 !important;
                color: inherit !important;
                user-select: none !important;
                display: flex !important;
                align-items: center !important;
                background: var(--background-fill-secondary, rgba(128, 128, 128, 0.08)) !important;
            }
            .ref2va-raw-prompt-content {
                padding: 6px 8px !important;
            }

            /* Symmetrical Uniform Grid for All 6 Prompt Sections */
            .ref2va-grid-row {
                display: flex;
                gap: 8px;
                margin-bottom: 6px;
            }
            .ref2va-rich-field-wrapper {
                flex: 1 1 50%;
                display: flex;
                flex-direction: column;
                margin-bottom: 0px !important;
                gap: 2px !important;
                padding: 0 !important;
            }
            .ref2va-rich-field-wrapper > .gr-html {
                margin: 0 !important;
                padding: 0 !important;
            }
            .ref2va-field-label {
                font-size: 11.5px;
                font-weight: 700;
                color: inherit;
                margin: 0 0 2px 0 !important;
                padding: 0 !important;
                line-height: 1.2;
            }
            .ref2va-rich-editor {
                border: 1px solid var(--border-color-primary, rgba(128, 128, 128, 0.3));
                border-radius: 6px;
                padding: 8px 10px;
                background: var(--input-background-fill, rgba(15, 23, 42, 0.4));
                color: var(--body-text-color, #e2e8f0);
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 12.5px;
                line-height: 1.45;
                white-space: pre-wrap;
                word-wrap: break-word;
                outline: none;
                overflow-y: auto;
                box-sizing: border-box;
                transition: border-color 0.15s ease;
                height: 160px !important;
                min-height: 160px !important;
                max-height: 160px !important;
            }
            .ref2va-rich-editor:focus {
                border-color: #3b82f6;
                box-shadow: 0 0 0 1px #3b82f6;
            }

            .ref2va-hidden-gradio-input,
            .ref2va-hidden-gradio-input.gr-block,
            .ref2va-hidden-gradio-input.block {
                display: none !important;
                margin: 0 !important;
                padding: 0 !important;
                height: 0 !important;
                min-height: 0 !important;
            }

            .ref2va-inline-badge {
                border-radius: 4px;
                padding: 1px 6px;
                margin: 0 2px;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                gap: 3px;
                cursor: text;
                font-size: 11.5px;
                transition: filter 0.15s ease;
            }
            .ref2va-inline-badge:hover {
                filter: brightness(1.25);
            }
            .ref2va-badge-subject {
                background: rgba(59, 130, 246, 0.22);
                border: 1px solid rgba(59, 130, 246, 0.55);
                color: #60a5fa;
            }
            .ref2va-badge-subject::before { content: '👤 '; }

            .ref2va-badge-picture {
                background: rgba(16, 185, 129, 0.22);
                border: 1px solid rgba(16, 185, 129, 0.55);
                color: #34d399;
            }
            .ref2va-badge-picture::before { content: '🖼️ '; }

            .ref2va-badge-video {
                background: rgba(168, 85, 247, 0.22);
                border: 1px solid rgba(168, 85, 247, 0.55);
                color: #c084fc;
            }
            .ref2va-badge-video::before { content: '🎥 '; }

            .ref2va-badge-audio {
                background: rgba(245, 158, 11, 0.22);
                border: 1px solid rgba(245, 158, 11, 0.55);
                color: #fbbf24;
            }
            .ref2va-badge-audio::before { content: '🔊 '; }

            .ref2va-badge-speaker {
                background: rgba(6, 182, 212, 0.22);
                border: 1px solid rgba(6, 182, 212, 0.55);
                color: #22d3ee;
            }
            .ref2va-badge-speaker::before { content: '🗣️ '; }

            .ref2va-badge-dialogue {
                background: rgba(244, 63, 94, 0.2);
                border: 1px solid rgba(244, 63, 94, 0.5);
                color: #fb7185;
            }
            .ref2va-badge-dialogue::before { content: '💬 '; }

            .ref2va-badge-shot {
                background: rgba(239, 68, 68, 0.22);
                border: 1px solid rgba(239, 68, 68, 0.55);
                color: #f87171;
            }
            .ref2va-badge-shot::before { content: '🎬 '; }

            .ref2va-badge-trans {
                background: rgba(20, 184, 166, 0.22);
                border: 1px solid rgba(20, 184, 166, 0.5);
                color: #2dd4bf;
            }
            .ref2va-badge-trans::before { content: '✂️ '; }

            .ref2va-badge-retention {
                background: rgba(99, 102, 241, 0.22);
                border: 1px solid rgba(99, 102, 241, 0.5);
                color: #a5b4fc;
            }
            .ref2va-badge-retention::before { content: '📌 '; }

            .ref2va-badge-task {
                background: rgba(148, 163, 184, 0.2);
                border: 1px solid rgba(148, 163, 184, 0.45);
                color: #93c5fd;
            }
            .ref2va-badge-task::before { content: '🎯 '; }

            .ref2va-badge-generic {
                background: rgba(100, 116, 139, 0.2);
                border: 1px solid rgba(100, 116, 139, 0.45);
                color: #cbd5e1;
            }

            .ref2va-preview-card {
                position: absolute;
                z-index: 99999;
                background: #0f172a;
                border: 1px solid #3b82f6;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.75);
                border-radius: 8px;
                padding: 10px;
                width: 290px;
                max-width: 90vw;
                color: #f8fafc;
                font-family: inherit;
            }
            .ref2va-popup-header {
                font-size: 12px;
                color: #38bdf8;
                margin-bottom: 6px;
                padding-right: 20px;
            }
            .ref2va-popup-close-btn {
                position: absolute;
                top: 6px;
                right: 8px;
                background: transparent;
                border: none;
                color: #94a3b8;
                font-size: 16px;
                cursor: pointer;
                line-height: 1;
            }
            .ref2va-popup-close-btn:hover {
                color: #ffffff;
            }
            .ref2va-popup-body {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }
            .ref2va-preview-media {
                max-width: 100%;
                max-height: 180px;
                border-radius: 4px;
                object-fit: contain;
                background: #020617;
            }
            .ref2va-popup-desc {
                font-size: 11.5px;
                line-height: 1.4;
                color: #e2e8f0;
                background: rgba(30, 41, 59, 0.6);
                border-radius: 4px;
                padding: 6px 8px;
            }
            .ref2va-popup-muted {
                color: #94a3b8;
                font-size: 11.5px;
            }
            </style>
            """

            with gr.Column(visible=initial_visible, elem_classes=["ref2va-container"]) as ref2va_main_container:
                gr.HTML(custom_css)
                
                # Header Bar (Zero Whitespace)
                with gr.Row(elem_classes=["ref2va-header-bar"]):
                    gr.HTML("<div class='ref2va-header-title'>🎬 MiniMax H3 Ref2VA Prompt Studio</div>")
                    with gr.Row():
                        settings_toggle_btn = gr.Button("⚙️ Settings", size="sm", min_width=90)
                        auto_boilerplate_btn = gr.Button("⚡ Auto-Fill Definitions", size="sm", min_width=175)
                        load_example_btn = gr.Button("📋 Load Example", size="sm", min_width=120)
                        clear_all_btn = gr.Button("🧹 Clear All", size="sm", min_width=75)

                # Settings Panel State & Container
                settings_open_state = gr.State(False)
                with gr.Column(visible=False, elem_classes=["ref2va-settings-drawer"]) as settings_panel:
                    gr.Markdown("#### ⚙️ MiniMax Ref2VA Studio Settings")
                    with gr.Row():
                        setting_active_mode = gr.Radio(
                            choices=[
                                ("Text Labels (with Hover Preview)", "text"),
                                ("Graphic Previews (Interactive Thumbnail Cards)", "graphic"),
                            ],
                            value=loaded_settings.get("active_refs_display_mode", "text"),
                            label="Active Reference Assets Display Mode"
                        )
                        setting_card_size = gr.Dropdown(
                            choices=[("Small (95px)", "small"), ("Medium (120px)", "medium"), ("Large (150px)", "large")],
                            value=loaded_settings.get("graphic_card_size", "medium"),
                            label="Graphic Card Size"
                        )
                    with gr.Row():
                        setting_sync_combined = gr.Checkbox(
                            label="Sync from combined prompt",
                            value=loaded_settings.get("sync_from_combined_prompt", False),
                            info="Automatically parse and update separate section editors when the combined main prompt is edited"
                        )
                    settings_json_bridge = gr.Textbox(value=json.dumps(loaded_settings), visible=False, elem_classes=["ref2va-settings-json-input"])

                # 1. Full Tag Palette (Manual Insertions)
                with gr.Accordion("🏷️ Full Tag Palette (Manual Insertions)", open=False):
                    with gr.Row(elem_classes=["ref2va-toolbar-group"]):
                        gr.HTML("<span class='ref2va-tag-label'>Subjects:</span>")
                        for i in range(1, 6):
                            b = gr.Button(f"<Subject {i}>", size="sm", elem_classes=["ref2va-btn"])
                            b.click(fn=None, js=f"() => window.insertRef2VAText('<Subject {i}>')")

                    with gr.Row(elem_classes=["ref2va-toolbar-group"]):
                        gr.HTML("<span class='ref2va-tag-label'>Generic Tags:</span>")
                        for i in range(1, 7):
                            bp = gr.Button(f"🖼️ <Picture {i}>", size="sm", elem_classes=["ref2va-btn"])
                            bp.click(fn=None, js=f"() => window.insertRef2VAText('<Picture {i}>')")
                        for i in range(1, 4):
                            bv = gr.Button(f"🎥 <Video {i}>", size="sm", elem_classes=["ref2va-btn"])
                            bv.click(fn=None, js=f"() => window.insertRef2VAText('<Video {i}>')")
                        for i in range(1, 3):
                            ba = gr.Button(f"🔊 <Audio {i}>", size="sm", elem_classes=["ref2va-btn"])
                            ba.click(fn=None, js=f"() => window.insertRef2VAText('<Audio {i}>')")

                    with gr.Row(elem_classes=["ref2va-toolbar-group"]):
                        gr.HTML("<span class='ref2va-tag-label'>Shots & Dialogue:</span>")
                        btn_shot1 = gr.Button("[Shot 1]", size="sm", elem_classes=["ref2va-btn"])
                        btn_shot1.click(fn=None, js="() => window.insertRef2VAText('[Shot 1] ')")

                        btn_shot2 = gr.Button("[Shot 2] At 00:03.000", size="sm", elem_classes=["ref2va-btn"])
                        btn_shot2.click(fn=None, js="() => window.insertRef2VAText('[Shot 2] At 00:03.000, ')")

                        btn_s1 = gr.Button("(S1)", size="sm", elem_classes=["ref2va-btn"])
                        btn_s1.click(fn=None, js="() => window.insertRef2VAText('(S1)')")

                        btn_s2 = gr.Button("(S2)", size="sm", elem_classes=["ref2va-btn"])
                        btn_s2.click(fn=None, js="() => window.insertRef2VAText('(S2)')")

                        btn_diag_en = gr.Button("💬 Dialogue <d>[English]</d>", size="sm", elem_classes=["ref2va-btn"])
                        btn_diag_en.click(fn=None, js="() => window.insertRef2VAText('', true, '<d>[English] ', '</d>')")

                        btn_diag_zh = gr.Button("💬 Dialogue <d>[Chinese]</d>", size="sm", elem_classes=["ref2va-btn"])
                        btn_diag_zh.click(fn=None, js="() => window.insertRef2VAText('', true, '<d>[Chinese] ', '</d>')")

                        btn_scenetrans = gr.Button("<scenetrans>", size="sm", elem_classes=["ref2va-btn"])
                        btn_scenetrans.click(fn=None, js="() => window.insertRef2VAText('<scenetrans>')")

                        btn_cutoff = gr.Button("<cutoff>", size="sm", elem_classes=["ref2va-btn"])
                        btn_cutoff.click(fn=None, js="() => window.insertRef2VAText('<cutoff>')")

                    with gr.Row(elem_classes=["ref2va-toolbar-group"]):
                        gr.HTML("<span class='ref2va-tag-label'>Retention:</span>")
                        ret_markers = ["fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference", "fully_copy", "reference"]
                        for marker in ret_markers:
                            bm = gr.Button(marker, size="sm", elem_classes=["ref2va-btn"])
                            bm.click(fn=None, js=f"() => window.insertRef2VAText('{marker}')")

                    with gr.Row(elem_classes=["ref2va-toolbar-group"]):
                        gr.HTML("<span class='ref2va-tag-label'>Task Types:</span>")
                        task_markers = ["[reference generation]", "[keyframe completion]", "[video editing]", "[video continuation]", "[audio reuse]", "[audio reference]"]
                        for task in task_markers:
                            bt = gr.Button(task, size="sm", elem_classes=["ref2va-btn"])
                            bt.click(fn=None, js=f"() => window.insertRef2VAText('{task} ')")

                # 2. Dynamic Active Reference Assets Bar
                gr.HTML("<div class='ref2va-active-refs-mount'></div>")

                # 3. Symmetrical 6 Section Rich Editors Grid (Exact Same Size: 160px height)
                with gr.Row(elem_classes=["ref2va-grid-row"]):
                    with gr.Column(elem_classes=["ref2va-rich-field-wrapper", "ref2va-wrapper-subject_definitions"]):
                        gr.HTML(f"<div class='ref2va-field-label'>{SECTION_DISPLAY_NAMES['subject_definitions']}</div>")
                        sec_subject_defs = gr.Textbox(value="", lines=6, elem_classes=["ref2va-hidden-gradio-input"])
                    with gr.Column(elem_classes=["ref2va-rich-field-wrapper", "ref2va-wrapper-summary"]):
                        gr.HTML(f"<div class='ref2va-field-label'>{SECTION_DISPLAY_NAMES['summary']}</div>")
                        sec_summary = gr.Textbox(value="", lines=6, elem_classes=["ref2va-hidden-gradio-input"])

                with gr.Row(elem_classes=["ref2va-grid-row"]):
                    with gr.Column(elem_classes=["ref2va-rich-field-wrapper", "ref2va-wrapper-retention_analysis"]):
                        gr.HTML(f"<div class='ref2va-field-label'>{SECTION_DISPLAY_NAMES['retention_analysis']}</div>")
                        sec_retention = gr.Textbox(value="", lines=6, elem_classes=["ref2va-hidden-gradio-input"])
                    with gr.Column(elem_classes=["ref2va-rich-field-wrapper", "ref2va-wrapper-detailed_description"]):
                        gr.HTML(f"<div class='ref2va-field-label'>{SECTION_DISPLAY_NAMES['detailed_description']}</div>")
                        sec_detailed = gr.Textbox(value="", lines=6, elem_classes=["ref2va-hidden-gradio-input"])

                with gr.Row(elem_classes=["ref2va-grid-row"]):
                    with gr.Column(elem_classes=["ref2va-rich-field-wrapper", "ref2va-wrapper-overall_soundscape"]):
                        gr.HTML(f"<div class='ref2va-field-label'>{SECTION_DISPLAY_NAMES['overall_soundscape']}</div>")
                        sec_soundscape = gr.Textbox(value="", lines=6, elem_classes=["ref2va-hidden-gradio-input"])
                    with gr.Column(elem_classes=["ref2va-rich-field-wrapper", "ref2va-wrapper-non_diegetic_music"]):
                        gr.HTML(f"<div class='ref2va-field-label'>{SECTION_DISPLAY_NAMES['non_diegetic_music']}</div>")
                        sec_music = gr.Textbox(value="", lines=6, elem_classes=["ref2va-hidden-gradio-input"])

            # Settings Event Listeners
            def toggle_settings_drawer(is_open: bool):
                new_state = not is_open
                return new_state, gr.update(visible=new_state)

            settings_toggle_btn.click(
                fn=toggle_settings_drawer,
                inputs=[settings_open_state],
                outputs=[settings_open_state, settings_panel],
                show_progress="hidden"
            )

            def update_and_persist_settings(mode, size, sync_combined):
                new_settings = {
                    "active_refs_display_mode": mode,
                    "graphic_card_size": size,
                    "sync_from_combined_prompt": bool(sync_combined)
                }
                save_settings(new_settings)
                return json.dumps(new_settings)

            for comp in [setting_active_mode, setting_card_size, setting_sync_combined]:
                comp.change(
                    fn=update_and_persist_settings,
                    inputs=[setting_active_mode, setting_card_size, setting_sync_combined],
                    outputs=[settings_json_bridge],
                    show_progress="hidden"
                ).then(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js="() => window.refreshActiveReferencesBar()"
                )

            # Auto-generate boilerplate based on active references
            media_inputs = [
                self.image_start if self.image_start is not None else gr.State(None),
                self.image_refs if self.image_refs is not None else gr.State(None),
                self.image_end if self.image_end is not None else gr.State(None),
                self.video_guide if self.video_guide is not None else gr.State(None),
                self.video_source if self.video_source is not None else gr.State(None),
                self.audio_guide if self.audio_guide is not None else gr.State(None),
                self.audio_guide2 if self.audio_guide2 is not None else gr.State(None)
            ]

            active_media_components = [comp for comp in [
                self.image_start, self.image_refs, self.image_end,
                self.video_guide, self.video_source,
                self.audio_guide, self.audio_guide2
            ] if comp is not None]

            for comp in active_media_components:
                comp.change(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js="() => { setTimeout(window.refreshActiveReferencesBar, 150); }",
                    show_progress="hidden"
                )

            def generate_reference_boilerplate(img_start, img_refs, img_end, vid_guide, vid_src, aud_guide, aud_guide2):
                pic_idx = 0
                subj_defs = []
                ret_analyses = []
                tasks = []

                if img_start is not None and str(img_start).strip() != "":
                    pic_idx += 1
                    subj_defs.append(f"<Picture {pic_idx}> is the first frame anchor of [Shot 1].")
                    ret_analyses.append(f"<Picture {pic_idx}> ([Shot 1] first frame): fully_preserved - exact starting composition preserved.")
                    tasks.append("keyframe completion")

                if img_refs is not None:
                    count_refs = len(img_refs) if isinstance(img_refs, list) else 1
                    for idx in range(count_refs):
                        pic_idx += 1
                        subj_num = idx + 1
                        subj_defs.append(f"<Subject {subj_num}> is the character/content in <Picture {pic_idx}>, preserving identity and clothing.")
                        ret_analyses.append(f"<Subject {subj_num}> (appears in [Shot 1]): fully_preserved - key appearance features retained.")
                    tasks.append("reference generation")

                vid_idx = 0
                if vid_src is not None and str(vid_src).strip() != "":
                    vid_idx += 1
                    subj_defs.append(f"<Video {vid_idx}> is the source video continuation.")
                    tasks.append("video continuation")

                if vid_guide is not None and str(vid_guide).strip() != "":
                    vid_idx += 1
                    subj_defs.append(f"<Video {vid_idx}> provides motion / camera rhythm guidance.")
                    tasks.append("reference generation")

                aud_idx = 0
                if aud_guide is not None and str(aud_guide).strip() != "":
                    aud_idx += 1
                    subj_defs.append(f"<Audio {aud_idx}> is the voice-timbre and dialogue reference for <Subject 1> (S1).")
                    ret_analyses.append(f"<Audio {aud_idx}>: reference - guides delivery timbre without copying raw waveform.")
                    tasks.append("audio reference")

                unique_tasks = list(dict.fromkeys(tasks)) or ["reference generation"]
                task_prefix = f"[{' + '.join(unique_tasks)}]"
                summary_text = f"{task_prefix} The target video features the referenced subjects in a coherent single take."

                defs_out = "\n".join(subj_defs) if subj_defs else SECTION_PLACEHOLDERS["subject_definitions"]
                ret_out = "\n".join(ret_analyses) if ret_analyses else SECTION_PLACEHOLDERS["retention_analysis"]

                return defs_out, summary_text, ret_out

            auto_boilerplate_btn.click(
                fn=generate_reference_boilerplate,
                inputs=media_inputs,
                outputs=[sec_subject_defs, sec_summary, sec_retention]
            ).then(
                fn=assemble_multisection_prompt,
                inputs=[sec_subject_defs, sec_summary, sec_retention, sec_detailed, sec_soundscape, sec_music],
                outputs=self.main_prompt if self.main_prompt is not None else []
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                js="() => window.updateRichEditorsFromTextareas(true)"
            )

            # Synchronized Prompt Handling
            section_inputs = [sec_subject_defs, sec_summary, sec_retention, sec_detailed, sec_soundscape, sec_music]

            if self.main_prompt is not None:
                for sec in section_inputs:
                    sec.input(
                        fn=assemble_multisection_prompt,
                        inputs=section_inputs,
                        outputs=self.main_prompt,
                        show_progress="hidden"
                    )

            # Sync from combined prompt back to separate sections (controlled by setting)
            def on_external_main_prompt_change(raw_val, state, s_def, s_sum, s_ret, s_det, s_snd, s_mus):
                current_settings = load_settings()
                if not current_settings.get("sync_from_combined_prompt", False):
                    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                prompt_text = raw_val or ""
                if isinstance(state, dict):
                    model_type = state.get("model_type", "")
                    all_settings = state.get("all_settings", {})
                    if model_type in all_settings and all_settings[model_type].get("prompt"):
                        prompt_text = all_settings[model_type].get("prompt")
                    elif state.get("prompt"):
                        prompt_text = state.get("prompt")

                current_assembled = assemble_multisection_prompt(s_def, s_sum, s_ret, s_det, s_snd, s_mus)
                if prompt_text.strip() == current_assembled.strip():
                    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                parsed = parse_multisection_prompt(prompt_text)
                return (
                    parsed["subject_definitions"],
                    parsed["summary"],
                    parsed["retention_analysis"],
                    parsed["detailed_description"],
                    parsed["overall_soundscape"],
                    parsed["non_diegetic_music"],
                )

            # Explicit file/form load parser
            def on_form_reload_or_file_load(raw_val, state, s_def, s_sum, s_ret, s_det, s_snd, s_mus):
                prompt_text = raw_val or ""
                if isinstance(state, dict):
                    model_type = state.get("model_type", "")
                    all_settings = state.get("all_settings", {})
                    if model_type in all_settings and all_settings[model_type].get("prompt"):
                        prompt_text = all_settings[model_type].get("prompt")
                    elif state.get("prompt"):
                        prompt_text = state.get("prompt")

                parsed = parse_multisection_prompt(prompt_text)
                return (
                    parsed["subject_definitions"],
                    parsed["summary"],
                    parsed["retention_analysis"],
                    parsed["detailed_description"],
                    parsed["overall_soundscape"],
                    parsed["non_diegetic_music"],
                )

            if self.main_prompt is not None:
                self.main_prompt.change(
                    fn=on_external_main_prompt_change,
                    inputs=[self.main_prompt, self.state_component if self.state_component is not None else gr.State(None)] + section_inputs,
                    outputs=section_inputs,
                    show_progress="hidden"
                ).then(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js="() => { window.updateRichEditorsFromTextareas(false); window.refreshActiveReferencesBar(); }"
                )

            if self.refresh_form_trigger is not None:
                self.refresh_form_trigger.change(
                    fn=on_form_reload_or_file_load,
                    inputs=[self.main_prompt if self.main_prompt is not None else gr.State(""), self.state_component if self.state_component is not None else gr.State(None)] + section_inputs,
                    outputs=section_inputs,
                    show_progress="hidden"
                ).then(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js="() => { setTimeout(() => { window.updateRichEditorsFromTextareas(true); window.refreshActiveReferencesBar(); }, 200); }"
                )

            if self.settings_file is not None:
                self.settings_file.upload(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js="() => { setTimeout(() => { window.refreshActiveReferencesBar(); }, 500); }",
                    show_progress="hidden"
                )

            def load_example():
                assembled = assemble_multisection_prompt(
                    EXAMPLE_REWRITE["subject_definitions"],
                    EXAMPLE_REWRITE["summary"],
                    EXAMPLE_REWRITE["retention_analysis"],
                    EXAMPLE_REWRITE["detailed_description"],
                    EXAMPLE_REWRITE["overall_soundscape"],
                    EXAMPLE_REWRITE["non_diegetic_music"]
                )
                return (
                    EXAMPLE_REWRITE["subject_definitions"],
                    EXAMPLE_REWRITE["summary"],
                    EXAMPLE_REWRITE["retention_analysis"],
                    EXAMPLE_REWRITE["detailed_description"],
                    EXAMPLE_REWRITE["overall_soundscape"],
                    EXAMPLE_REWRITE["non_diegetic_music"],
                    assembled
                )

            load_example_btn.click(
                fn=load_example,
                inputs=None,
                outputs=section_inputs + ([self.main_prompt] if self.main_prompt is not None else [])
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                js="() => window.updateRichEditorsFromTextareas(true)"
            )

            def clear_all():
                return ["", "", "", "", "", "", ""]

            clear_all_btn.click(
                fn=clear_all,
                inputs=None,
                outputs=section_inputs + ([self.main_prompt] if self.main_prompt is not None else [])
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                js="""() => {
                    const secNames = ['subject_definitions', 'summary', 'retention_analysis', 'detailed_description', 'overall_soundscape', 'non_diegetic_music'];
                    secNames.forEach(secName => {
                        const richEditor = document.querySelector(`.ref2va-editor-${secName}`);
                        if (richEditor) richEditor.innerHTML = '<div><br></div>';
                    });
                }"""
            )

            # Model switch listener
            def on_model_target_change(target_val):
                model_name = str(target_val or "").split("|")[0].strip()
                base_type = ""
                if hasattr(self, "get_base_model_type") and callable(self.get_base_model_type):
                    base_type = self.get_base_model_type(model_name) or ""
                
                is_active = is_minimax_model(model_name, base_type)
                return gr.update(visible=is_active)

            if self.model_choice_target is not None:
                self.model_choice_target.change(
                    fn=on_model_target_change,
                    inputs=[self.model_choice_target],
                    outputs=[ref2va_main_container],
                    show_progress="hidden"
                ).then(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js="() => { window.wrapMainPromptInAccordion(); window.setupAllRichEditors(); setTimeout(window.refreshActiveReferencesBar, 250); }"
                )

            self.ref2va_container = ref2va_main_container
            return ref2va_main_container

        target_id = "prompt_column_advanced" if self.prompt_column_advanced is not None else "prompt"
        self.insert_after(
            target_component_id=target_id,
            new_component_constructor=create_studio_ui
        )

    def on_model_change(self, state: Dict[str, Any], model_type: str):
        base_type = ""
        if hasattr(self, "get_base_model_type") and callable(self.get_base_model_type):
            base_type = self.get_base_model_type(model_type) or ""

        is_target_model = is_minimax_model(model_type, base_type)
        return gr.update(visible=is_target_model)