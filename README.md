# Polus

A powerful tensorflow toolkit for creating/training complex deep learning models in a functional way.

This toolkit is currently under development and aims to focus on biomedical tasks, although it can also also be used in other domains.

### Installation

```
pip install polus
```

For training the models in multiGPU scenario, polus leverage the horovod library for dataparalelism, which means that horovod must be
correctly installed to be able to run the training in multiGPUs.

To easy this process we made available a docker image with all the correct dependencies to perform the multiGPU training in polus.

```
docker pull talmeidawastaken/polus
```

### Documentation

Still in work, but some of it can be already consulted here: https://bioinformatics-ua.github.io/polus/

### Team
  * Tiago Almeida<sup id="a1">[1](#f1)</sup>
  * Rui Antunes<sup id="a1">[1](#f1)</sup>
  * Sérgio Matos<sup id="a1">[1](#f1)</sup>

1. <small id="f1"> University of Aveiro, Dept. Electronics, Telecommunications and Informatics (DETI / IEETA), Aveiro, Portugal </small>