
_visiontext() {
    local cur prev opts
    _init_completion || return
    # complete first argument with script
    if [ $COMP_CWORD -eq 1 ]; then
        opts="bboxes cacheutils colormaps font htmltools images imports iotools.feature_compression iotools.lmdbext iotools.tar_indexer iotools.tar_lookup nlp.lemmatizer nlp.nltktools nlp.regextools pandatools plots profiling.code_profiler profiling.hardware_profiler run.profile_gpu spacytools"
        COMPREPLY=( $( compgen -W "${opts}" -- "${cur}") )
        return 0
    fi
    # otherwise complete with filesystem
    COMPREPLY=( $(compgen -f -- "${cur}") )
    return 0
}

complete -F _visiontext visiontext
