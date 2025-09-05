import boto3, os, json, tarfile
from afqmc import classical, quantum


def setup_and_run():
    """
    This method runs the user code.
    """
    print("Running container script")
    s3_client = boto3.client("s3")

    job_id = os.getenv("AWS_BATCH_JOB_ID")
    print(f"Job ID: {job_id}")

    s3_bucket_name = os.getenv("JOB_S3_BUCKET_NAME")
    output_file_name = os.getenv("JOB_OUTPUT_FILE_NAME")

    entry_point = os.getenv("JOB_ENTRY_POINT")

    array_index = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", default="-1"))
    print(f"Array index: {array_index}")

    # Run algorithm
    result = {}
    if entry_point == "run_classical_afqmc":
        result = classical.run(int(os.getenv('JOB_TIME_STEPS')), float(os.getenv('JOB_DTAU')))
    elif entry_point == "run_qc_afqmc":
        input_file_key = os.getenv("JOB_INPUT_FILE_KEY")
        s3_client.download_file(s3_bucket_name, input_file_key, "model.tar.gz")
        tar = tarfile.open('model.tar.gz', 'r:gz')
        tar.extractall('inputs')
        tar.close()
        print("The matchgate shadows are successfully retrieved from S3.")

        result = quantum.run(int(os.getenv('JOB_TIME_STEPS')), float(os.getenv('JOB_DTAU')))

    # Save result to output
    output_path = os.path.join(os.getcwd(), output_file_name)
    print(f"Save result to output path: {output_path}")
    with open(output_path, "w") as f:
        json.dump(result, f)

    # Upload results to S3
    s3_key = os.path.join("batch", job_id, output_file_name)
    print(f"Upload output to S3: {s3_key}")
    s3_client.upload_file(output_file_name, s3_bucket_name, s3_key)
    print("Job completed.")


if __name__ == "__main__":
    setup_and_run()
