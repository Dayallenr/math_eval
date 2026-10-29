FROM python:3.12-slim
RUN pip install --no-cache-dir "math-verify[antlr4_13_2]==0.9.0" "latex2sympy2-extended==1.11.0" "sympy==1.14.0"
WORKDIR /app
