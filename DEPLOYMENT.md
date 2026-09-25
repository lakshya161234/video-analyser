# Deploy ClipMind

ClipMind can be deployed on Streamlit Community Cloud from this GitHub repository.

## Before deploying

1. Push the project changes to the GitHub repository you want to deploy.
2. Keep `.env` and `.streamlit/secrets.toml` out of Git. `.gitignore` excludes both.
3. Create a fresh API key for the LLM provider you plan to use. For Hinglish transcription, also create a Sarvam API key.

## Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
2. Select this repository, its branch, and `app.py` as the app file.
3. In **Advanced settings**, select Python 3.11.
4. Add secrets using `.streamlit/secrets.toml.example` as a template. Replace the example values with real keys. For example:

   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your-new-groq-key"
   SARVAM_API_KEY = "your-sarvam-key"
   ```

   Only add the keys for providers you use. `LLM_PROVIDER` can be `groq`, `mistral`, or `gemini`. Gemini uses `GOOGLE_API_KEY`; Mistral uses `MISTRAL_API_KEY`. `SARVAM_API_KEY` is only needed for Hinglish transcription.

5. Click **Deploy**. Streamlit installs packages from the root `requirements.txt` and uses `.streamlit/config.toml` for the app theme.

## Runtime notes

- English transcription runs OpenAI Whisper locally. Its default model is `small`; on a memory-limited host, set `WHISPER_MODEL = "base"` or `"tiny"` in the host secrets. Smaller models use fewer resources but may be less accurate.
- Hinglish transcription uses Sarvam's API and requires `SARVAM_API_KEY`.
- YouTube processing requires the host to allow outbound access to YouTube. Uploaded recordings are also supported.
- Uploaded media and intermediate audio are temporary. Meeting results live in the active app session and are not saved as a user library.
- The app sends transcript content to the selected LLM provider to create notes and answer questions.
