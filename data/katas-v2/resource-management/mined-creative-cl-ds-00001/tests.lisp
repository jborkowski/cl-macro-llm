;;; tests.lisp — generated from j14i/cl-ds row 1
;;; instruction: Write a Common Lisp macro `with-mutex-held` that acquires a lock via `acquire-lock`, executes the body, and guarantees r
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-mutex-held (*lock*) (incf *counter*)) . (LET ((#:G1 *LOCK*)) (ACQUIRE-LOCK #:G1) (UNWIND-PROTECT (PROGN (INCF *COUNTER*)) (RELEASE-LOCK #:G1)))))
