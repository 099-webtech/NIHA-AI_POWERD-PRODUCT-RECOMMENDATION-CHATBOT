// ChatSideMenu.js
import React, { useRef, useState, useEffect } from 'react';
import ChatHistory from './ChatHistory';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars } from '@fortawesome/free-solid-svg-icons';
import axios from 'axios'; // Import axios for making API calls
import Cookies from 'js-cookie'; // Import Cookies for getting username from cookies

function ChatSideMenu({ onNewChat, onChatSelect }) {
    const [isMenuVisible, setMenuVisible] = useState(false);
    const myDivRef = useRef(null);
    const [groups, setGroups] = useState([]); // New state for groups
    const [dropdownOpen, setDropdownOpen] = useState(false); // State to manage dropdown visibility
    const userName = Cookies.get('userName'); // Get the username from cookies
    const [chats, setChats] = useState([]); // State for chat history

    // Fetch groups and chats from the database when the component mounts
    useEffect(() => {
        const fetchGroupsAndChats = async () => {
            try {
                const groupsResponse = await axios.get(`http://localhost:2000/groups/${userName}`);
                setGroups(groupsResponse.data);

                const chatsResponse = await axios.get(`http://localhost:2000/chats/${userName}`);
                setChats(chatsResponse.data); // Set the chats state with the fetched data
            } catch (error) {
                console.error("Error fetching data:", error);
            }
        };

        fetchGroupsAndChats();
    }, [userName]); // Dependency array includes userName to refetch if it changes

    const hiddenVisibleToggle = () => {
        if (myDivRef.current) {
            myDivRef.current.classList.toggle('hidden', isMenuVisible);
            myDivRef.current.classList.toggle('visible', !isMenuVisible);
        }
        setMenuVisible(!isMenuVisible);
    };

    // Function to create a new group
    const createGroup = async () => {
        const groupName = prompt('Enter group name:');
        if (groupName) {
            try {
                const existingGroupsResponse = await axios.get(`http://localhost:2000/groups/${userName}`);
                const existingGroups = existingGroupsResponse.data;

                const groupExists = existingGroups.some(group => group.name === groupName);
                if (groupExists) {
                    alert('Group name already exists. Please choose a different name.'); // Alert if group exists
                    return;
                }

                const response = await axios.post(`http://localhost:2000/groups/${userName}`, {
                    name: groupName,
                    username: userName
                });
                setGroups([...groups, { ...response.data, chats: [] }]); // Initialize chats as an empty array
            } catch (error) {
                console.error("Error creating group:", error);
            }
        }
    };

    // Function to rename a group
    const renameGroup = async (groupId) => {
        const newName = prompt('Enter new group name:');
        if (newName) {
            try {
                // Check if the new group name already exists
                const existingGroupsResponse = await axios.get(`http://localhost:2000/groups/${userName}`);
                const existingGroups = existingGroupsResponse.data;

                const groupExists = existingGroups.some(group => group.name === newName);
                if (groupExists) {
                    alert('Group name already exists. Please choose a different name.');
                    return;
                }

                // Send PUT request to the server to rename the group
                await axios.put(`http://localhost:2000/groups/${userName}/${groupId}`, {
                    name: newName
                });

                // Update local state to reflect the renamed group
                setGroups(groups.map(group => group._id === groupId ? { ...group, name: newName } : group));
            } catch (error) {
                console.error("Error renaming group:", error);
            }
        }
    };

    // Function to delete a group
    const deleteGroup = async (groupId) => {
        if (window.confirm('Are you sure you want to delete this group?')) {
            try {
                // Send DELETE request to the server to delete the group
                await axios.delete(`http://localhost:2000/groups/${userName}/${groupId}`);
                
                // Update the local state to remove the deleted group
                setGroups(groups.filter(group => group._id !== groupId));
            } catch (error) {
                console.error("Error deleting group:", error);
            }
        }
    };

    return (
        <>
            <div className={`sidemenu-toggle-btn ${isMenuVisible ? 'visible' : 'hidden'}`} onClick={hiddenVisibleToggle}>
                <FontAwesomeIcon icon={faBars} />
            </div>
            <div className={`chat-sidemenu ${isMenuVisible ? 'visible' : 'hidden'}`} ref={myDivRef}>
                <div className="chat-sidemenu-content">
                    <div className="sidemenu-header">
                        <button className="new_chat_button" onClick={onNewChat}>
                            <img src="/static/images/plus.png" alt="add" style={{ width: '25px', marginRight: '10px' }} />
                            New Chat
                        </button>
                        {/* <button className="new_chat_button" onClick={createGroup}>
                            <img src="/static/images/plus.png" alt="add" style={{ width: '25px', marginRight: '10px' }} />
                            New Group
                        </button> */}
                    </div>
                    {/* <div className="groupContainer">
                        <button className="group_new_chat_button" onClick={() => setDropdownOpen(!dropdownOpen)}>
                            Groups
                            <img src="/static/images/dropdrop.png" alt="dropdown" style={{ width: '25px', marginLeft: '5px' }} />
                        </button>
                        {dropdownOpen && (
                            <ul className="groupDropdown">
                                {groups.map(group => (
                                    <li 
                                        key={group._id} 
                                        style={{ padding: '5px', margin: '5px 0' }} // Optional styling
                                    >
                                        {group.name}
                                        <button onClick={() => renameGroup(group._id)}>Rename</button>
                                        <button onClick={() => deleteGroup(group._id)}>Delete</button>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div> */}
                    <ChatHistory onChatSelect={onChatSelect} groups={groups} chats={chats} setChats={setChats} />
                </div>
            </div>
        </>
    );
}

export default ChatSideMenu;
