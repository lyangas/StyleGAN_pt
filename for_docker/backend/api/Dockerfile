FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

ADD ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ADD . .

EXPOSE 5080

ENTRYPOINT ["python", "-u", "gen_image_api.py"]