;;; tests.lisp — generated from j14i/cl-ds row 262
;;; instruction: Write a Common Lisp macro `with-pid-file` that writes a process id stamp to a file at the given path, runs body, and del
;;; category: resource-management | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((with-pid-file "/tmp/app.pid" (serve)) . (LET ((#:G1 "/tmp/app.pid")) (WITH-OPEN-FILE (#:G2 #:G1 :DIRECTION :OUTPUT :IF-EXISTS :SUPERSEDE :IF-DOES-NOT-EXIST :CREATE) (FORMAT #:G2 "~A" (RANDOM 1000000))) (UNWIND-PROTECT (PROGN (SERVE)) (WHEN (PROBE-FILE #:G1) (DELETE-FILE #:G1))))))
