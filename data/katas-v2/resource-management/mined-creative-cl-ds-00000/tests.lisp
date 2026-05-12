;;; tests.lisp — generated from j14i/cl-ds row 0
;;; instruction: Write a Common Lisp macro `with-temp-file` that binds a variable to a temporary file path, executes the body, and delete
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-temp-file (f "/tmp/foo.txt") (write-stuff f)) . (LET* ((#:G1 "/tmp/foo.txt") (F #:G1)) (UNWIND-PROTECT (PROGN (WRITE-STUFF F)) (WHEN (PROBE-FILE #:G1) (DELETE-FILE #:G1))))))
