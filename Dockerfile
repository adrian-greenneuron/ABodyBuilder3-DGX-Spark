ARG BASE_IMAGE=abodybuilder3-dev-spark:optimised-251026
FROM ${BASE_IMAGE}
COPY . /formatter
RUN sh /formatter/tools/build_native.sh \
    && python3 -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'abb3_formatter_experiment.pth').write_text('/formatter\nimport native_activation\n')"
# Neither the installed model nor the writer changes without this explicit flag:
# docker run -e ABB3_NATIVE_FORMATTER=1 ...
ENV ABB3_NATIVE_FORMATTER=0
