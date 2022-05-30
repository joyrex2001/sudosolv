#################
## Build image ## ------------------------------------------------------------
#################

FROM docker.io/golang:1.18 AS build

ARG CODE=github.com/joyrex2001/sudosolv

RUN apt update && apt install -y libopencv-dev

ADD . /go/src/${CODE}/
RUN cd /go/src/${CODE} \
    && go build -o sudosolv \
    && mkdir /app \
    && cp sudosolv /app

#################
## Final image ## ------------------------------------------------------------
#################

FROM debian:bullseye-slim

RUN apt update && apt install -y libopencv-dev
      
COPY --from=build /app /app

COPY trained.bin /app

WORKDIR /app

ENTRYPOINT ["/app/sudosolv"]
CMD [ "server" ]
