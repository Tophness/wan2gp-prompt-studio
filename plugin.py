import gradio as gr
import re
import json
from typing import Dict, Any, List, Tuple
from shared.utils.plugins import WAN2GPPlugin

TARGET_MODEL_ARCHITECTURE = "minimax_h3_ref2va"

# Canonical prompt sections in required order
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
        self.version = "1.2.0"
        self.description = "Full-Reference structured prompt generator with live reference media tracking for MiniMax H3 Ref2VA."
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
        self.request_component("model_choice_target")
        self.request_component("state")
        self.request_global("get_base_model_type")
        self.request_global("get_model_def")
        self.request_global("server_config")

        # Client-side cursor-aware insertion into whichever section textarea is active
        self.add_custom_js("""
        (function() {
            window._lastFocusedRef2VATextarea = null;

            document.addEventListener('focusin', function(e) {
                if (e.target && e.target.tagName === 'TEXTAREA' && e.target.closest('.ref2va-textarea-container')) {
                    window._lastFocusedRef2VATextarea = e.target;
                }
            });

            window.insertRef2VAText = function(textToInsert, isWrap = false, wrapPrefix = '', wrapSuffix = '') {
                let textarea = window._lastFocusedRef2VATextarea;
                if (!textarea || !document.body.contains(textarea)) {
                    textarea = document.querySelector('.ref2va-detailed-desc textarea') || document.querySelector('.ref2va-textarea-container textarea');
                }

                if (!textarea) return;

                textarea.focus();
                const start = textarea.selectionStart || 0;
                const end = textarea.selectionEnd || 0;
                const selectedText = textarea.value.substring(start, end);
                let replacement = textToInsert;

                if (isWrap) {
                    replacement = wrapPrefix + (selectedText || '') + wrapSuffix;
                }

                const newText = textarea.value.substring(0, start) + replacement + textarea.value.substring(end);
                
                const proto = window.HTMLTextAreaElement.prototype;
                const nativeValueSetter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                nativeValueSetter.call(textarea, newText);

                const event = new Event('input', { bubbles: true });
                textarea.dispatchEvent(event);

                const newCursorPos = start + (isWrap && !selectedText ? wrapPrefix.length : replacement.length);
                textarea.setSelectionRange(newCursorPos, newCursorPos);
            };
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
        self.model_choice_target = components.get("model_choice_target")
        self.state_component = components.get("state")

        initial_model = ""
        if hasattr(self, "server_config") and isinstance(self.server_config, dict):
            initial_model = self.server_config.get("last_model_type", "")
        
        initial_visible = is_minimax_model(initial_model)

        def create_studio_ui():
            custom_css = """
            <style>
            .ref2va-container {
                border: 1px solid rgba(59, 130, 246, 0.35);
                background: transparent;
                border-radius: 8px;
                padding: 12px 14px;
                margin-top: 10px;
                margin-bottom: 14px;
            }
            .ref2va-active-refs-bar {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                align-items: center;
                margin-top: 6px;
                margin-bottom: 10px;
            }
            .ref2va-toolbar-group {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-bottom: 6px;
                align-items: center;
            }
            .ref2va-btn {
                font-size: 11px !important;
                padding: 3px 8px !important;
                min-width: unset !important;
                height: 26px !important;
            }
            .ref2va-btn-ref {
                background: rgba(16, 185, 129, 0.15) !important;
                border: 1px solid rgba(16, 185, 129, 0.45) !important;
                color: #10b981 !important;
                font-weight: 600 !important;
            }
            .ref2va-tag-label {
                font-size: 11px;
                font-weight: bold;
                color: #3b82f6;
                margin-right: 4px;
            }
            </style>
            """

            with gr.Column(visible=initial_visible, elem_classes=["ref2va-container"]) as ref2va_main_container:
                gr.HTML(custom_css)
                
                # Header & Action Controls
                with gr.Row():
                    gr.Markdown("### 🎬 MiniMax H3 Ref2VA Prompt Studio")
                    with gr.Row():
                        auto_boilerplate_btn = gr.Button("⚡ Auto-Fill Definitions from References", size="sm", min_width=210)
                        load_example_btn = gr.Button("📋 Load Example Template", size="sm", min_width=140)
                        parse_from_main_btn = gr.Button("🔄 Parse from Main Prompt", size="sm", min_width=140)
                        clear_all_btn = gr.Button("🧹 Clear All", size="sm", min_width=75)

                # 1. Full Tag Palette / Manual Insertions (Top, Collapsed by default)
                with gr.Accordion("🏷️ Full Tag Palette (Manual Insertions)", open=False):
                    with gr.Row(elem_classes=["ref2va-toolbar-group"]):
                        gr.HTML("<span class='ref2va-tag-label'>Subjects:</span>")
                        for i in range(1, 6):
                            b = gr.Button(f"<Subject {i}>", size="sm", elem_classes=["ref2va-btn"])
                            b.click(fn=None, js=f"() => window.insertRef2VAText('<Subject {i}>')")

                    with gr.Row(elem_classes=["ref2va-toolbar-group"]):
                        gr.HTML("<span class='ref2va-tag-label'>Generic Tags:</span>")
                        for i in range(1, 5):
                            bp = gr.Button(f"🖼️ <Picture {i}>", size="sm", elem_classes=["ref2va-btn"])
                            bp.click(fn=None, js=f"() => window.insertRef2VAText('<Picture {i}>')")
                        for i in range(1, 3):
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

                # 2. Dynamic Active Reference Assets Bar (Directly Below, Clean & Live)
                with gr.Row(elem_classes=["ref2va-active-refs-bar"], visible=False) as active_refs_row:
                    gr.HTML("<span class='ref2va-tag-label' style='color:#10b981;'>⚡ Active Reference Assets:</span>")
                    dyn_btn_pic1 = gr.Button("🖼️ <Picture 1>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])
                    dyn_btn_pic2 = gr.Button("🖼️ <Picture 2>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])
                    dyn_btn_pic3 = gr.Button("🖼️ <Picture 3>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])
                    dyn_btn_pic4 = gr.Button("🖼️ <Picture 4>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])
                    dyn_btn_vid1 = gr.Button("🎥 <Video 1>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])
                    dyn_btn_vid2 = gr.Button("🎥 <Video 2>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])
                    dyn_btn_aud1 = gr.Button("🔊 <Audio 1>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])
                    dyn_btn_aud2 = gr.Button("🔊 <Audio 2>", size="sm", visible=False, elem_classes=["ref2va-btn", "ref2va-btn-ref"])

                    dyn_btn_pic1.click(fn=None, js="() => window.insertRef2VAText('<Picture 1>')")
                    dyn_btn_pic2.click(fn=None, js="() => window.insertRef2VAText('<Picture 2>')")
                    dyn_btn_pic3.click(fn=None, js="() => window.insertRef2VAText('<Picture 3>')")
                    dyn_btn_pic4.click(fn=None, js="() => window.insertRef2VAText('<Picture 4>')")
                    dyn_btn_vid1.click(fn=None, js="() => window.insertRef2VAText('<Video 1>')")
                    dyn_btn_vid2.click(fn=None, js="() => window.insertRef2VAText('<Video 2>')")
                    dyn_btn_aud1.click(fn=None, js="() => window.insertRef2VAText('<Audio 1>')")
                    dyn_btn_aud2.click(fn=None, js="() => window.insertRef2VAText('<Audio 2>')")

                # The 6 Dedicated Sections
                with gr.Row():
                    with gr.Column():
                        sec_subject_defs = gr.Textbox(
                            label=SECTION_DISPLAY_NAMES["subject_definitions"],
                            placeholder=SECTION_PLACEHOLDERS["subject_definitions"],
                            lines=4,
                            elem_classes=["ref2va-textarea-container"]
                        )
                    with gr.Column():
                        sec_summary = gr.Textbox(
                            label=SECTION_DISPLAY_NAMES["summary"],
                            placeholder=SECTION_PLACEHOLDERS["summary"],
                            lines=4,
                            elem_classes=["ref2va-textarea-container"]
                        )

                with gr.Row():
                    with gr.Column():
                        sec_retention = gr.Textbox(
                            label=SECTION_DISPLAY_NAMES["retention_analysis"],
                            placeholder=SECTION_PLACEHOLDERS["retention_analysis"],
                            lines=4,
                            elem_classes=["ref2va-textarea-container"]
                        )
                    with gr.Column():
                        sec_detailed = gr.Textbox(
                            label=SECTION_DISPLAY_NAMES["detailed_description"],
                            placeholder=SECTION_PLACEHOLDERS["detailed_description"],
                            lines=7,
                            elem_classes=["ref2va-textarea-container", "ref2va-detailed-desc"]
                        )

                with gr.Row():
                    with gr.Column():
                        sec_soundscape = gr.Textbox(
                            label=SECTION_DISPLAY_NAMES["overall_soundscape"],
                            placeholder=SECTION_PLACEHOLDERS["overall_soundscape"],
                            lines=2,
                            elem_classes=["ref2va-textarea-container"]
                        )
                    with gr.Column():
                        sec_music = gr.Textbox(
                            label=SECTION_DISPLAY_NAMES["non_diegetic_music"],
                            placeholder=SECTION_PLACEHOLDERS["non_diegetic_music"],
                            lines=2,
                            elem_classes=["ref2va-textarea-container"]
                        )

            # --- Live Reference Asset Inspection (Triggered automatically on media change) ---

            def scan_references(img_start, img_refs, img_end, vid_guide, vid_src, aud_guide, aud_guide2):
                pic_count = 0
                vid_count = 0
                aud_count = 0

                # 1. Images
                if img_start is not None:
                    pic_count += 1
                
                if img_refs is not None:
                    count_refs = len(img_refs) if isinstance(img_refs, list) else 1
                    pic_count += count_refs

                if img_end is not None:
                    pic_count += 1

                # 2. Videos
                if vid_src is not None:
                    vid_count += 1

                if vid_guide is not None:
                    vid_count += 1

                # 3. Audios
                if aud_guide is not None:
                    aud_count += 1

                if aud_guide2 is not None:
                    aud_count += 1

                has_any_ref = (pic_count + vid_count + aud_count) > 0

                return (
                    gr.update(visible=has_any_ref),
                    gr.update(visible=pic_count >= 1),
                    gr.update(visible=pic_count >= 2),
                    gr.update(visible=pic_count >= 3),
                    gr.update(visible=pic_count >= 4),
                    gr.update(visible=vid_count >= 1),
                    gr.update(visible=vid_count >= 2),
                    gr.update(visible=aud_count >= 1),
                    gr.update(visible=aud_count >= 2),
                )

            media_inputs = [
                self.image_start if self.image_start is not None else gr.State(None),
                self.image_refs if self.image_refs is not None else gr.State(None),
                self.image_end if self.image_end is not None else gr.State(None),
                self.video_guide if self.video_guide is not None else gr.State(None),
                self.video_source if self.video_source is not None else gr.State(None),
                self.audio_guide if self.audio_guide is not None else gr.State(None),
                self.audio_guide2 if self.audio_guide2 is not None else gr.State(None)
            ]

            dyn_btn_outputs = [
                active_refs_row,
                dyn_btn_pic1, dyn_btn_pic2, dyn_btn_pic3, dyn_btn_pic4,
                dyn_btn_vid1, dyn_btn_vid2,
                dyn_btn_aud1, dyn_btn_aud2
            ]

            # Automatically hook change events on all media input components
            active_media_components = [comp for comp in [
                self.image_start, self.image_refs, self.image_end,
                self.video_guide, self.video_source,
                self.audio_guide, self.audio_guide2
            ] if comp is not None]

            for comp in active_media_components:
                comp.change(
                    fn=scan_references,
                    inputs=media_inputs,
                    outputs=dyn_btn_outputs,
                    show_progress="hidden"
                )

            # Auto-generate boilerplate based on active references
            def generate_reference_boilerplate(img_start, img_refs, img_end, vid_guide, vid_src, aud_guide, aud_guide2):
                pic_idx = 0
                subj_defs = []
                ret_analyses = []
                tasks = []

                if img_start is not None:
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
                if vid_src is not None:
                    vid_idx += 1
                    subj_defs.append(f"<Video {vid_idx}> is the source video continuation.")
                    tasks.append("video continuation")

                if vid_guide is not None:
                    vid_idx += 1
                    subj_defs.append(f"<Video {vid_idx}> provides motion / camera rhythm guidance.")
                    tasks.append("reference generation")

                aud_idx = 0
                if aud_guide is not None:
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
            )

            # Auto-sync back to single prompt in WanGP
            section_inputs = [sec_subject_defs, sec_summary, sec_retention, sec_detailed, sec_soundscape, sec_music]

            if self.main_prompt is not None:
                for sec in section_inputs:
                    sec.input(
                        fn=assemble_multisection_prompt,
                        inputs=section_inputs,
                        outputs=self.main_prompt,
                        show_progress="hidden"
                    )

            def do_parse(raw):
                parsed = parse_multisection_prompt(raw)
                return [parsed[k] for k in SECTION_KEYS]

            if self.main_prompt is not None:
                parse_from_main_btn.click(
                    fn=do_parse,
                    inputs=self.main_prompt,
                    outputs=section_inputs
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
                outputs=section_inputs + [self.main_prompt]
            )

            def clear_all():
                return ["", "", "", "", "", "", ""]

            clear_all_btn.click(
                fn=clear_all,
                inputs=None,
                outputs=section_inputs + [self.main_prompt]
            )

            # Dynamic model switch visibility listener
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