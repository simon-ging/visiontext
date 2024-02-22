
_visiontext() {
    local cur prev opts
    _init_completion || return
    # complete first argument with script
    if [ $COMP_CWORD -eq 1 ]; then
        opts="profile_gpu"
        COMPREPLY=( $( compgen -W "${opts}" -- "${cur}") )
        return 0
    fi
    # otherwise complete with filesystem
    COMPREPLY=( $(compgen -f -- "${cur}") )
    return 0
}

complete -F _visiontext visiontext
