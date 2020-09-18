#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import sys

json_file_name = sys.argv[1]
img_list_name = sys.argv[2]

json_text = None
with open(json_file_name, 'r') as f:
    json_text = f.read()

matched_list = re.findall( r'\"([0-9]+.jpg)\"', json_text)

with open(img_list_name, 'w') as f:
    for img_name in matched_list:
        f.write(img_name)
        f.write('\n')
