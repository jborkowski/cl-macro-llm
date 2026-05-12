;;; tests.lisp — generated from j14i/cl-ds row 378
;;; instruction: Write a Common Lisp macro `unless-equal` that runs body forms only when two expressions `a` and `b` are not EQUAL. Each 
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((unless-equal (left) (right) (warn "diff")) . (LET ((#:G1 (LEFT)) (#:G2 (RIGHT))) (UNLESS (EQUAL #:G1 #:G2) (WARN "diff")))))
