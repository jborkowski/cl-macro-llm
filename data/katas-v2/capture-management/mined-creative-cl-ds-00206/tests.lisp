;;; tests.lisp — generated from j14i/cl-ds row 206
;;; instruction: Write a Common Lisp macro `with-locked` that acquires a lock expression (evaluating it only once), runs body inside unwi
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((with-locked (my-lock) (work)) . (LET ((#:G1 MY-LOCK)) (ACQUIRE-LOCK #:G1) (UNWIND-PROTECT (PROGN (WORK)) (RELEASE-LOCK #:G1)))))
