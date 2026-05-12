;;; tests.lisp — hand-crafted from On Lisp anaphoric family
;;; instruction: Write `aor` — anaphoric or. Bind the primary form's
;;; result to IT so the default branch can reference what came back.
;;; category: capture-management | technique: anaphoric | complexity: basic
;;; quality_score: 0.9

(
 ((aor (lookup-cache key) (compute-default))
  . (LET ((IT (LOOKUP-CACHE KEY))) (OR IT (COMPUTE-DEFAULT))))

 ((aor (get-env "PORT") 8080)
  . (LET ((IT (GET-ENV "PORT"))) (OR IT 8080)))

 ((aor (gethash :name table) :anonymous)
  . (LET ((IT (GETHASH :NAME TABLE))) (OR IT :ANONYMOUS)))
)
