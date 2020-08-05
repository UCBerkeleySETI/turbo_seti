coverage run --source=turbo_seti -m pytest
EXITCODE=$?
if [ $EXITCODE -ne 0 ]; then
    echo
    echo '*** Oops, coverage pytest failed, exit code = '$EXITCODE' ***'
    echo
    exit $EXITCODE
fi
coverage report
EXITCODE=$?
if [ $EXITCODE -ne 0 ]; then
    echo
    echo '*** Oops, coverage report failed, exit code = '$EXITCODE' ***'
    echo
    exit $EXITCODE
fi
codecov
