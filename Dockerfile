FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install diffusers torch torchvision transformers accelerate flask

EXPOSE 5000

CMD ["python", "app.py"]