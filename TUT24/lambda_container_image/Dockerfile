FROM public.ecr.aws/lambda/python:3.10

ARG SCRIPT_PATH=lambda_container_image
COPY ${SCRIPT_PATH}/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip3 install -r requirements.txt

COPY ${SCRIPT_PATH}/index.py ${LAMBDA_TASK_ROOT}/

CMD [ "index.handler" ]