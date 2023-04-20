FROM minkowski_engine

WORKDIR workdir/

RUN pip install h5py

COPY *py ./
#COPY process_hits.py .
#COPY test_docker.py .
#COPY 


#CMD ["python", "test_docker.py", "data/mini.h5"]
#CMD ["python", "train.py", "--trainsample", "data/mini.h5", "--validatesample", "data/mini.h5", "--batch_size", "2", "--epochs", "1", "--noweight", "--output_dir", "output"]
#ENTRYPOINT ["python", "./train.py"]
#CMD ["pwd"]
