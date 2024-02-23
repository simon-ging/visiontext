
_filedir()  # source: ubuntu 2004 /usr/share/bash-completion/bash_completion
{
    local IFS=$'\n'
    _tilde "$cur" || return
    local -a toks
    local reset
    if [[ "$1" == -d ]]; then
        reset=$(shopt -po noglob); set -o noglob
        toks=( $(compgen -d -- "$cur") )
        IFS=' '; $reset; IFS=$'\n'
    else
        local quoted
        _quote_readline_by_ref "$cur" quoted
        # Munge xspec to contain uppercase version too
        # http://thread.gmane.org/gmane.comp.shells.bash.bugs/15294/focus=15306
        local xspec=${1:+"!*.@($1|${1^^})"} plusdirs=()
        # Use plusdirs to get dir completions if we have a xspec; if we don't,
        # there's no need, dirs come along with other completions. Don't use
        # plusdirs quite yet if fallback is in use though, in order to not ruin
        # the fallback condition with the "plus" dirs.
        local opts=( -f -X "$xspec" )
        [[ $xspec ]] && plusdirs=(-o plusdirs)
        [[ ${COMP_FILEDIR_FALLBACK-} ]] || opts+=( "${plusdirs[@]}" )
        reset=$(shopt -po noglob); set -o noglob
        toks+=( $(compgen "${opts[@]}" -- $quoted) )
        IFS=' '; $reset; IFS=$'\n'
        # Try without filter if it failed to produce anything and configured to
        [[ -n ${COMP_FILEDIR_FALLBACK:-} && -n "$1" && ${#toks[@]} -lt 1 ]] && {
            reset=$(shopt -po noglob); set -o noglob
            toks+=( $(compgen -f "${plusdirs[@]}" -- $quoted) )
            IFS=' '; $reset; IFS=$'\n'
        }
    fi
    if [[ ${#toks[@]} -ne 0 ]]; then
        # 2>/dev/null for direct invocation, e.g. in the _filedir unit test
        compopt -o filenames 2>/dev/null
        COMPREPLY+=( "${toks[@]}" )
    fi
} # _filedir()



_visiontext() {
    local cur prev opts
    _init_completion || return
    # complete first argument with script
    if [ $COMP_CWORD -eq 1 ]; then
        opts="profile_gpu"
        COMPREPLY=( $( compgen -W "${opts}" -- "${cur}") )
        return 0
    fi
    # # otherwise complete with filesystem
    # COMPREPLY=( $(compgen -f -- "${cur}") )
    _filedir
    return 0
}

complete -F _visiontext visiontext
