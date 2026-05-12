;;; tests.lisp — generated from j14i/cl-ds row 267
;;; instruction: Define a macro that eliminates this repetitive pattern:
;;; category: capture-management | technique: once-only,gensym | complexity: intermediate
;;; quality_score: 0.875

(((ok nil "This supposed to be failed") . (let ((#:test150 nil) (#:desc151 "This supposed to be failed")) (with-catching-errors (:expected t :description #:desc151) (with-duration ((#:duration148 #:result149) #:test150) (test #:result149 t #:desc151 :duration #:duration148 :test-fn (lambda (x y) (eq (not (null x)) y)) :got-form #:test150))))))
