mkdir -p ~/.streamlit/

echo "\
[API_main]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml
