gendoc:
	docker build -t trlxgendocs -f docker/docs/Dockerfile .
run:
	docker run --rm -it \
		-p 8000:8000 \
		--entrypoint python trlxgendocs -m http.server 8000 --directory build/docs/build/html

sh:
	docker run --rm -it \
		-p 8000:8000 \
		--entrypoint /bin/bash trlxgendocs
