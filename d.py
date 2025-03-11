import serial

# Initialize the serial port
ser = serial.Serial('COM9', 9600)  # Adjust 'COM1' to your serial port and 9600 to your baud rate

# Convert integer 1 to string and encode it
data_to_send = str(1).encode()

# Send the data over the serial port
ser.write(data_to_send)

# Close the serial port when done
ser.close()