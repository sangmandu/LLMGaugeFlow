FROM python:3

ENV PYTHONUNBUFFERED=1
ENV PYTHONMALLOC=malloc

COPY . .

RUN pip install --upgrade pip --no-cache-dir || /bin/bash
RUN pip install --no-cache-dir -r requirements.txt || /bin/bash

CMD ["python", "app.py"]
