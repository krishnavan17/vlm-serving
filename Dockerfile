FROM python:3.12-slim

WORKDIR /app

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=${http_proxy} \
	https_proxy=${https_proxy} \
	no_proxy=${no_proxy} \
	HTTP_PROXY=${HTTP_PROXY} \
	HTTPS_PROXY=${HTTPS_PROXY} \
	NO_PROXY=${NO_PROXY}

COPY vlm-serving/requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r /app/requirement.txt

COPY vlm-serving /app/vlm-serving

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "/app/vlm-serving", "--host", "0.0.0.0", "--port", "8000"]
