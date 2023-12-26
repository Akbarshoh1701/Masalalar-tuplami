
"""1-misol

# class Solution:
#     def defangIPaddr(self, address: str) -> str:
#         return address.replace(".", "[.]")
# # """


#2-misol
"""class Solution:
    def buildArray(self, nums: list[int]) -> list[int]:
        new_list = []
        for i in nums:
            new_list.append(nums[i])

        return new_list
"""

#3 mispl
"""class Solution:
    def finalValueAfterOperations(self, operations: list[str]) -> int:
        X = 0
        for i in operations:
            if i == "--X" or i == "X--":
                X -= 1
            elif i == "X++" or "++X":
                X += 1

        return X

"""

#4-misol
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        resalt = ""
        bir, ikki = len(word1), len(word2)
        nev_list = min(bir, ikki)

        for i in range(nev_list):
            resalt += word1[1] + word2[i]

        resalt += word1[nev_list:] + word2[nev_list:]





