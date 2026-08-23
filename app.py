import streamlit as st
from pathlib import Path
from PIL import Image

from src.detector import (
    load_detector,
    detect_image,
)

from src.metadata_analysis import (
    create_metadata_report,
    analyze_metadata_with_deepseek,
    model_ai_score,
    combine_scores,
    final_label,
    choose_generator,
)


@st.cache_resource
def get_detector():
    return load_detector()


st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🤖",
    layout="centered",
)


st.title("AI Image Detector")

st.write(
    "This app checks the image pixels and its metadata "
    "before giving a final result."
)

st.caption(
    "Only the metadata report is sent to DeepSeek. "
    "The uploaded image itself is not sent to DeepSeek."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        st.success("Image uploaded")

        st.image(
            image,
            caption=uploaded_file.name,
        )


        if st.button("Confirm"):

            with st.spinner(
                "Running image and metadata checks..."
            ):

                # -----------------------------
                # CHECK 1:
                # Hugging Face image detector
                # -----------------------------

                detector = get_detector()

                label, score = detect_image(
                    detector,
                    image,
                )

                pixel_score = model_ai_score(
                    label,
                    score,
                )


                # -----------------------------
                # CHECK 2:
                # Create metadata report
                # -----------------------------

                report, report_text = (
                    create_metadata_report(
                        image,
                        uploaded_file.name,
                    )
                )


                # -----------------------------
                # Save metadata report
                # directly inside project folder
                # -----------------------------

                project_folder = (
                    Path(__file__)
                    .resolve()
                    .parent
                )

                metadata_folder = (
                    project_folder
                    / "metadata_reports"
                )

                metadata_folder.mkdir(
                    exist_ok=True
                )

                metadata_path = (
                    metadata_folder
                    / "metadata_report.json"
                )

                metadata_path.write_text(
                    report_text,
                    encoding="utf-8",
                )


                # Save report in Streamlit memory
                # so it stays visible after reruns

                st.session_state[
                    "metadata_report"
                ] = report_text

                st.session_state[
                    "metadata_path"
                ] = str(
                    metadata_path.resolve()
                )


                # -----------------------------
                # DeepSeek metadata analysis
                # -----------------------------

                metadata_result = {
                    "usable_for_ai_detection": False,
                    "ai_probability": None,
                    "generator": "Unknown",
                    "evidence_strength": "none",
                    "explanation": (
                        "DeepSeek metadata analysis "
                        "was not available."
                    ),
                    "evidence": [],
                }

                metadata_score = None


                try:

                    api_key = st.secrets[
                        "DEEPSEEK_API_KEY"
                    ]

                    (
                        metadata_result,
                        metadata_score,
                    ) = analyze_metadata_with_deepseek(
                        report_text,
                        api_key,
                    )


                except Exception as e:

                    metadata_result[
                        "explanation"
                    ] = (
                        "DeepSeek metadata check "
                        f"was skipped: {e}"
                    )


                # -----------------------------
                # Combine the scores
                # -----------------------------

                evidence_strength = str(
                    metadata_result.get(
                        "evidence_strength",
                        "none",
                    )
                ).lower()


                final_score = combine_scores(
                    pixel_score,
                    metadata_score,
                    evidence_strength,
                )


                result = final_label(
                    final_score
                )


                generator = choose_generator(
                    metadata_result,
                    final_score,
                )


                # -----------------------------
                # Save results in session state
                # -----------------------------

                st.session_state[
                    "analysis_done"
                ] = True

                st.session_state[
                    "final_result"
                ] = result

                st.session_state[
                    "final_score"
                ] = final_score

                st.session_state[
                    "pixel_score"
                ] = pixel_score

                st.session_state[
                    "metadata_score"
                ] = metadata_score

                st.session_state[
                    "metadata_result"
                ] = metadata_result

                st.session_state[
                    "generator"
                ] = generator


        # =================================
        # DISPLAY FINAL RESULTS
        # =================================

        if st.session_state.get(
            "analysis_done",
            False,
        ):

            result = st.session_state[
                "final_result"
            ]

            final_score = st.session_state[
                "final_score"
            ]

            pixel_score = st.session_state[
                "pixel_score"
            ]

            metadata_score = st.session_state[
                "metadata_score"
            ]

            metadata_result = st.session_state[
                "metadata_result"
            ]

            generator = st.session_state[
                "generator"
            ]


            st.divider()

            st.subheader(
                "Final Result"
            )


            if result == "Likely AI-generated":

                st.warning(
                    result
                )


            elif result == "Likely real":

                st.success(
                    result
                )


            else:

                st.info(
                    result
                )


            st.metric(
                "AI likelihood",
                f"{final_score * 100:.1f}%"
            )


            st.write(
                "**Possible generator / family:** "
                + generator
            )


            st.subheader(
                "Why?"
            )


            st.write(
                "**Hugging Face image model:** "
                f"{pixel_score * 100:.1f}% AI score"
            )


            st.write(
                "**DeepSeek metadata analysis:** "
                + str(
                    metadata_result.get(
                        "explanation",
                        "No explanation returned.",
                    )
                )
            )


            if metadata_score is not None:

                st.write(
                    "**Metadata AI score:** "
                    f"{metadata_score * 100:.1f}%"
                )


            else:

                st.write(
                    "**Metadata AI score:** "
                    "Not used in the calculation"
                )


            # -----------------------------
            # Show metadata evidence
            # -----------------------------

            evidence = metadata_result.get(
                "evidence",
                [],
            )


            if evidence:

                st.write(
                    "**Metadata evidence:**"
                )

                for item in evidence:

                    st.write(
                        f"- {item}"
                    )


        # =================================
        # DISPLAY METADATA REPORT
        # =================================

        if (
            "metadata_report"
            in st.session_state
        ):

            st.divider()

            st.subheader(
                "Metadata Report"
            )


            st.success(
                "Metadata report created successfully!"
            )


            st.write(
                "**Saved locally at:**"
            )

            st.code(
                st.session_state[
                    "metadata_path"
                ]
            )


            with st.expander(
                "View metadata report"
            ):

                st.code(
                    st.session_state[
                        "metadata_report"
                    ],
                    language="json",
                )


            st.download_button(
                label="Download metadata report",
                data=st.session_state[
                    "metadata_report"
                ],
                file_name="metadata_report.json",
                mime="application/json",
            )


            st.caption(
                "The metadata report above is the same "
                "information that is sent to DeepSeek "
                "for analysis."
            )


        st.caption(
            "This result is an estimate, not proof. "
            "Metadata can be removed or edited, "
            "and AI detectors can make mistakes."
        )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )


else:

    st.info(
        "Upload an image to get started."
    )

