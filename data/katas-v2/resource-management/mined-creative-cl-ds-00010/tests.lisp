;;; tests.lisp — generated from j14i/cl-ds row 10
;;; instruction: Write a Common Lisp macro `with-temporary-directory` that ensures a directory exists, binds a user variable to its pathn
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-temporary-directory (d #p"/tmp/work/") (process d)) . (LET* ((#:G1 #P"/tmp/work/") (D #:G1)) (ENSURE-DIRECTORIES-EXIST #:G1) (UNWIND-PROTECT (PROGN (PROCESS D)) (DELETE-DIRECTORY #:G1)))))
