// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.4.17;

contract CropFeedback {
    struct Feedback {
        string name;
        string place;
        string email;
        string feedback;
    }

    Feedback[] public feedbacks;

    // Function to store feedback
    function storeFeedback(string memory _name, string memory _place, string memory _email, string memory _feedback) public {
        feedbacks.push(Feedback(_name, _place, _email, _feedback));
    }

    // Function to retrieve feedback by index
    function getFeedback(uint256 _index) public view returns (string memory, string memory, string memory, string memory) {
       // require(_index < feedbacks.length, "Feedback index out of bounds");
        Feedback memory f = feedbacks[_index];
        return (f.name, f.place, f.email, f.feedback);
    }

    // Function to get the number of feedback entries
    function getFeedbackCount() public view returns (uint256) {
        return feedbacks.length;
    }

}


