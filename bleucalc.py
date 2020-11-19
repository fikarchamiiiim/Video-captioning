from nltk.translate.bleu_score import sentence_bleu

# reference1 = [['a','man','is','talking','about','a','website']]
# candidate1 = ['a','man','is','talking','about','a','website']
# score1 = sentence_bleu(reference1, candidate1)
# print("video8996 - {}".format(score1))

# reference2 = [['a','man','is','playing','guitar']]
# candidate2 = ['a','man','is','playing','the','guitar']
# score2 = sentence_bleu(reference2, candidate2)
# print("video9165 - {}".format(score2))

# reference3 = [['a','man','is','showing','how','to','use','a','phone']]
# candidate3 = ['a','man','is','showing','how','to','use','a','phone']
# score3 = sentence_bleu(reference3, candidate3)
# print("video9389 - {}".format(score3))

# reference4 = [['a','man','is','talking','about','the','football','team']]
# candidate4 = ['a','man','is','talking','about','a','football','team']
# score4 = sentence_bleu(reference4, candidate4)
# print("video8944 - {}".format(score4))

# reference5 = [['a','cartoon','character','is','standing']]
# candidate5 = ['a','cartoon','character','is','standing']
# score5 = sentence_bleu(reference5, candidate5)
# print("video8933 - {}".format(score5))

# reference6 = [['a','person','is','solving','a','piece','of','electronic']]
# candidate6 = ['a','person','is','solving','a','piece','of','electronics']
# score6 = sentence_bleu(reference6, candidate6)
# print("video9411 - {}".format(score6))

# reference7 = [['a','man','is','playing','a','basketball','game']]
# candidate7 = ['a','man','is','playing','a','basketball','game']
# score7 = sentence_bleu(reference7, candidate7)
# print("video9018 - {}".format(score7))

# reference8 = [['a','cartoon','character','is','dancing']]
# candidate8 = ['a','cartoon','character','is','dancing']
# score8 = sentence_bleu(reference8, candidate8)
# print("video8794 - {}".format(score8))

# reference9 = [['the','crowd','of','applaused','people']]
# candidate9 = ['a','crowd','cheers','for','a','crowd']
# score9 = sentence_bleu(reference9, candidate9)
# print("video9893 - {}".format(score9))

reference0 = [['a','man','in','suit','is','talking','to','a','man']]
candidate0 = ['a','man','in','a','suit','is','talking','to','a','man']
score0 = sentence_bleu(reference0, candidate0)
print("video9532 - {}".format(score0))


