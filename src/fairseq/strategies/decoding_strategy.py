# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


class DecodingStrategy(object):
    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        pass
