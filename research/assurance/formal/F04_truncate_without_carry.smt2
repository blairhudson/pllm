(set-option :timeout 10000)

(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (not (= (bvadd ((_ extract 7 4) a) ((_ extract 7 4) b)) ((_ extract 7 4) (bvadd a b)))))
(check-sat)
(get-value (a b))
