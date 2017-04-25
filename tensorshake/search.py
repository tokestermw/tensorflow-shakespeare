from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq

START_ID = 2
END_ID = 3

MAXLEN = 20


class Beam(object):
    # For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    # This is so that if two prefixes have equal probabilities then a complete sentence is
    # preferred over an incomplete one since (0.5, False) < (0.5, True)

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


def beam_search(probabilities_function, beam_width=10, clip_len=MAXLEN, length_penalty=0.6):
    """ Use log probs. 
    """
    prev_beam = Beam(beam_width)
    prev_beam.add(0.0, False, [START_ID])

    while True:
        current_beam = Beam(beam_width)

        # Add complete sentences that do not yet have the best probability to the current beam,
        # the rest prepare to add more words to them.
        for (prefix_prob, complete, prefix) in prev_beam:
            # print(prefix)
            if complete:
                current_beam.add(prefix_prob, True, prefix)
            else:
                # Get probability of each possible next word for the incomplete prefix.
                for (next_prob, next_word) in probabilities_function(prefix):
                    if next_word == END_ID:
                        current_beam.add(prefix_prob + next_prob, True, prefix + [next_word])
                    else:  # if next word is a non-end token then mark prefix as incomplete
                        current_beam.add(prefix_prob + next_prob, False, prefix + [next_word])

        # print(current_beam.heap)
        (best_prob, best_complete, best_prefix) = max(current_beam)

        # if best_complete or len(
                # if most probable prefix is a complete sentence or has a length that exceeds the clip length
                # (ignoring the start token) then return it
                # best_prefix) - 1 == clip_len:
            # TODO: return all from heap
            # return (best_prefix[1:],
                    # best_prob)  # return best sentence without the start token and together with its probability

        all_complete = all([i[1] for i in current_beam.heap])
        if all_complete or len(best_prefix) - 1 == clip_len:
            return [i[2][1:] for i in sorted(current_beam.heap, key=lambda x: x[0], reverse=True)]

        prev_beam = current_beam


def length_penalty(sequence_length, penalty_factor=0.6):
    """Calculates the length penalty according to
    https://arxiv.org/abs/1609.08144
     """
    # return tf.div((5. + tf.to_float(sequence_lengths))**penalty_factor, (5. + 1.)
                                # **penalty_factor)
    return (5. + sequence_length) ** penalty_factor / ((5. + 1.) ** penalty_factor)


def _test():
    # config = BeamSearchConfig(
    #     beam_width=5,
    #     vocab_size=10,
    #     eos_token=3,
    #     length_penalty_weight=.1,
    #     choose_successors_fn=choose_top_k)
    # return BeamSearchDecoder(decoder=decoder, config=config)

    import random

    def prob_f(prefix):
        u = random.random()

        if u < .2:
            return [(.2, "ahoy"), (.2, "ahoy"), (.2, "ahoy"), (.2, "ahoy"), (.2, "ahoy")]
        elif u > .2 and u < .23:
            return [(.3, "<end>"), (.2, "ahoy"), (.2, "ahoy"), (.2, "ahoy"), (.2, "ahoy")]
        else:
            return [(.4, "doink"), (.2, "ahoy"), (.2, "ahoy"), (.2, "ahoy"), (.2, "ahoy")]

    print(beam_search(prob_f))


if __name__ == "__main__":
    _test()
