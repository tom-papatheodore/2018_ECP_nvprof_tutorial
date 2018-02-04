#!/bin/bash

module swap xl pgi/17.10
module load cuda

alias ls="ls --color=auto"

export PS1="\[\033[38;5;8m\][\W]\\$ \[$(tput sgr0)\]"
