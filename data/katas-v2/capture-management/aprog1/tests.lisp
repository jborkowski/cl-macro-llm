;;; tests.lisp — hand-crafted from On Lisp ch. 14
;;; instruction: Write `aprog1` — like PROG1 but binds the first form to
;;; IT so subsequent forms can reference it; returns IT.
;;; category: capture-management | technique: anaphoric | complexity: basic
;;; quality_score: 0.95

(
 ((aprog1 (make-instance 'session) (log :created it))
  . (LET ((IT (MAKE-INSTANCE 'SESSION)))
      (LOG :CREATED IT)
      IT))

 ((aprog1 (allocate-buffer 1024) (register it) (zero-fill it))
  . (LET ((IT (ALLOCATE-BUFFER 1024)))
      (REGISTER IT)
      (ZERO-FILL IT)
      IT))

 ((aprog1 (compute-result) (audit-log it))
  . (LET ((IT (COMPUTE-RESULT)))
      (AUDIT-LOG IT)
      IT))
)
