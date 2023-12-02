
_visiontext() {
    local cur prev opts
    _init_completion || return
    # complete first argument with script
    if [ $COMP_CWORD -eq 1 ]; then
        opts="images.image_to_html images.scale_image imports iotools.feature_compression iotools.images iotools.tar_indexer iotools.tar_lookup notebookutils.html_output panda.pandas_tables panda.printing run.profile_gpu visualizer.colormaps"
        COMPREPLY=( $( compgen -W "${opts}" -- "${cur}") )
        return 0
    fi
    # otherwise complete with filesystem
    COMPREPLY=( $(compgen -f -- "${cur}") )
    return 0
}

complete -F _visiontext visiontext
