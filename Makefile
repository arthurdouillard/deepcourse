
all: build

build:
	/usr/local/bin/hugo-0.79.0/hugo
	git add docs/
	git add -u
	git add static/code/ static/notebooks
	git commit -m "Update website"
	git push origin master

watch:
	/usr/local/bin/hugo-0.79.0/hugo serve --watch
